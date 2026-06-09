# 1. Install required libraries (run this cell once)
!pip install google-adk google-generativeai --quiet



from google.adk.agents import Agent
import google.generativeai as genai
import random



genai.configure(api_key="AIzaSyBl9hfQ-gw0vYAcaKH8hSGxGmRrB6KpHgw")
gemini_model = genai.GenerativeModel("models/gemini-2.5-pro")




class PlannerAgent(Agent):
    def plan_learning(self, goal):
        tasks = self.break_into_tasks(goal)
        schedule = self.create_weekly_schedule(tasks)
        return schedule

    def break_into_tasks(self, goal):
        response = gemini_model.generate_content(
            f"Break the learning goal '{goal}' into 5 tasks."
        )
        return [
            line for line in response.candidates[0].content.parts[0].text.split('\n') 
            if line.strip()
        ]

    def create_weekly_schedule(self, tasks):
        schedule = {f"Day_{i+1}": task for i, task in enumerate(tasks)}
        return schedule

class TeacherAgent(Agent):
    def teach(self, topic):
        notes = gemini_model.generate_content(f"Create detailed study notes for: {topic}")
        flashcards = gemini_model.generate_content(f"Make flashcards for: {topic}")
        summary = gemini_model.generate_content(f"Summarize: {topic}")
        return {
            "notes": notes.candidates[0].content.parts[0].text,
            "flashcards": flashcards.candidates[0].content.parts[0].text,
            "summary": summary.candidates[0].content.parts[0].text
        }

class EvaluatorAgent(Agent):
    def create_quiz(self, topic):
        questions = gemini_model.generate_content(
            f"Create 3 quiz questions with answers for: {topic}"
        )
        return [
            line for line in questions.candidates[0].content.parts[0].text.split('\n\n')
            if line.strip()
        ]

    def grade(self, user_answers, correct_answers):
        score = sum([
            ua.strip().lower() == ca.strip().lower()
            for ua, ca in zip(user_answers, correct_answers)
        ])
        return score



planner = PlannerAgent(name="planner")
teacher = TeacherAgent(name="teacher")
evaluator = EvaluatorAgent(name="evaluator")



goal = "Learn the basics of Machine Learning"



schedule = planner.plan_learning(goal)
print("Study Schedule:\n", schedule)



for day, topic in schedule.items():
    print(f"\n--- {day} ---")
    # Generate learning materials
    materials = teacher.teach(topic)
    print("Notes:\n", materials["notes"])
    print("Flashcards:\n", materials["flashcards"])
    print("Summary:\n", materials["summary"])
    
    # Generate a quiz
    quiz = evaluator.create_quiz(topic)
    print("Quiz Questions:\n", quiz)
    
    # Placeholder: simulate answers (all blank for demo)
    correct_answers = [""] * len(quiz)   # (If you parse/generate real answers, use them here)
    user_answers = [""] * len(quiz)      # (For interactive, use input() or widgets)
    score = evaluator.grade(user_answers, correct_answers)
    print(f"Score for {day}: {score}/{len(quiz)}")


