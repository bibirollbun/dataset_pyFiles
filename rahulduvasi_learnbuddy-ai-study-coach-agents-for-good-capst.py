# Kaggle already has the required libraries installed.
print("Environment ready â€“ no installation needed.")



import sys
sys.path.append("/kaggle/input/learnbuddy-src/learnbuddy-kaggle-upload/src")

from agents.intake import IntakeAgent
from agents.planner import PlannerAgent
from agents.content_agent import ContentAgent
from agents.quiz_agent import QuizAgent
from agents.feedback_agent import FeedbackAgent
from memory.memory_store import MemoryStore



memory = MemoryStore('/kaggle/working/students.json')

intake = IntakeAgent(memory)
planner = PlannerAgent(memory)
content = ContentAgent(memory)
quiz = QuizAgent(memory)
feedback = FeedbackAgent(memory)



# Step 1 â€” Student intake (simulated)
student = intake.collect_profile_simulated()
student



plan = planner.create_plan(student)
plan



lessons = content.generate_lessons(plan)
lessons



questions = quiz.generate_quiz(lessons)
questions



answers = quiz.simulate_answers(questions)
score = quiz.grade(questions, answers)
score



feedback_text = feedback.summarize_feedback(score, questions, answers)
feedback_text



memory.update_progress(student['id'], score, feedback_text)



# Full Pipeline Runner

student = intake.collect_profile_simulated()
plan = planner.create_plan(student)
lessons = content.generate_lessons(plan)
questions = quiz.generate_quiz(lessons)
answers = quiz.simulate_answers(questions)
score = quiz.grade(questions, answers)
feedback_text = feedback.summarize_feedback(score, questions, answers)
memory.update_progress(student["id"], score, feedback_text)

print("Pipeline executed successfully!\n")
print("Student:", student)
print("Plan:", plan)
print("Lessons:", lessons)
print("Questions:", questions)
print("Score:", score)
print("Feedback:", feedback_text)


