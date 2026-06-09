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


# --- 1. Sessions & Memory Feature ---

class SessionData:
    """
    Implements Sessions & Memory: Tracks state, including mastery and analysis data.
    """
    def __init__(self, user_name, topic_list):
        self.user_name = user_name
        self.current_topic = None
        self.mastery_threshold = 0.75
        self.performance_history = {t: 0.3 for t in topic_list} # Initial mastery for all topics
        self.topic_weightage = {} # Stores analysis results
        self.socratic_dialogue = []
        self.exam_notes = "" # Stores structured content

    def update_mastery(self, topic, score_change):
        """Updates the Mastery Score for a specific topic."""
        current = self.performance_history[topic]
        self.performance_history[topic] = min(1.0, max(0.0, current + score_change))
        print(f"[MEM] Mastery of '{topic}' updated to {self.performance_history[topic]*100:.0f}%")


# --- Knowledge Base and Simulated Data ---
KNOWLEDGE_BASE = {
    "Microbial Genetics": {"pyq_data": [9, 10, 7], "raw_text": "Includes topics on bacterial conjugation, transformation, transduction, and operon models."},
    "Enzyme Kinetics": {"pyq_data": [4, 6, 5], "raw_text": "Covers Michaelis-Menten kinetics, Lineweaver-Burk plots, and various types of inhibition."}
}


# --- Simulated User Input ---
SIMULATED_USER_RESPONSES = [
    "The equation is $V_0 = V_{max} \cdot [S] / (K_m + [S])$. It helps design bioreactors.", # Socratic Turn 1
    "It affects the initial rate, but $V_{max}$ is only affected by the total enzyme concentration, so it stays the same.", # Socratic Turn 2
    "Statement A and C are correct.", # Assessment Q1
]
input_index = 0
def get_simulated_input():
    global input_index
    if input_index < len(SIMULATED_USER_RESPONSES):
        response = SIMULATED_USER_RESPONSES[input_index]
        input_index += 1
        return response
    return "Skipping."


# --- 2. Tools Feature (LLM Interface) ---

class LLM_Tool:
    """Simulates LLM interaction for generation and analysis."""
    
    # Tool for Socratic Agent
    def generate_socratic_question(self, topic, dialogue_history):
        return f"AI: Can you explain the central equation of '{topic}' and its real-world relevance in biotechnology?" if not dialogue_history else f"AI: Now, explain *why* changing the substrate concentration affects the $V_{{max}}$."

    # Tool for Assessment Agent
    def generate_practice_test(self, topic):
        questions = [f"Q1 (MSQ): Which statements are TRUE regarding the principles of {topic}?"]
        return questions, {"Q1": "Statement A and C are correct."}
    
    # Tool for Socratic/Assessment Agents
    def evaluate_response(self, response):
        if "equation" in response.lower() or "why" in response.lower():
            return 0.3, "Good depth!"
        return -0.1, "Needs clarification."


# --- 3. Multi-Agent System & Extended Features ---


# Feature 4: Analytics Agent (PYQ Analysis)
class AnalyticsAgent:
    """Agent 1: Analyzes PYQ data to find topic weightage for strategic planning."""
    def analyze_pyq_weightage(self, topic_data):
        avg_marks = sum(topic_data["pyq_data"]) / len(topic_data["pyq_data"])
        weightage_score = min(1.0, avg_marks / 10.0) # Normalize
        return weightage_score


# Feature 5: Resource Agent (Content Ingestion)
class ResourceAgent:
    """Agent 2: Ingests raw data and generates structured Exam Notes (Custom Tool)."""
    def __init__(self, session): 
        self.session = session
        self.llm = LLM_Tool()
        # ---------------------------------------
    def generate_exam_notes(self, session: SessionData, raw_text):
        # Simulation of LLM summarizing raw_text into structured notes
        notes = f"## Exam Notes: {session.current_topic}\n* **PYQ Focus:** High/Medium Weightage.\n* **Key Formula:** $V_0 = V_{max} \cdot [S] / (K_m + [S])$.\n* **Concepts:** Substrate affects $V_0$, Enzyme concentration affects $V_{max}$."
        session.exam_notes = notes
        print(f"[RESOURCE AGENT] Notes generated and saved to memory for {session.current_topic}.")

class SocraticTutorAgent:
    """Agent 3 (AI Tutor): Implements the Loop Agent using the LLM_Tool."""
    def __init__(self, session: SessionData):
        self.session = session
        self.llm = LLM_Tool()
        self.turn = 0

    def run_socratic_loop(self, user_input):
        self.turn += 1
        
        question = self.llm.generate_socratic_question(self.session.current_topic, self.session.socratic_dialogue)
        print(f"\n--- Turn {self.turn} ---")
        print(question)
        print(f"YOU (Simulated): {user_input}")

        score_change, feedback = self.llm.evaluate_response(user_input)
        print(f"FEEDBACK: {feedback}")
        
        self.session.update_mastery(self.session.current_topic, score_change)
        
        if self.session.performance_history[self.session.current_topic] >= self.session.mastery_threshold:
            return "COMPLETED"
        return "CONTINUE"


class AssessmentAgent:
    """Agent 4: Provides Practice Tests (Demonstrates LLM-powered practice)."""
    def __init__(self, session: SessionData):
        self.session = session
        self.llm = LLM_Tool()

    def run_assessment(self):
        questions, answer_key = self.llm.generate_practice_test(self.session.current_topic)
        user_answer = get_simulated_input() # Get final simulated answer
        
        # Simple grading logic
        if user_answer == answer_key["Q1"]:
            print(f"\n[ASSESSMENT] Correct! Score 1/1.")
            self.session.update_mastery(self.session.current_topic, 0.2) # Reward for correct test answer
        else:
            print(f"\n[ASSESSMENT] Incorrect. The answer is '{answer_key['Q1']}'.")
            # Uses the Resource Agent's notes (stored in memory) for contextual feedback
            print(f"**AI TUTOR TIP:** Revisit Notes: {self.session.exam_notes.split('## Exam Notes:')[1]}")
            self.session.update_mastery(self.session.current_topic, -0.1) # Penalty for incorrect test answer
        
        print("--- Assessment Complete ---")


class PlanningAgent:
    """Agent 5: Orchestrator that links all agents and features."""
    def __init__(self, session: SessionData):
        self.session = session
        self.analytics = AnalyticsAgent()
        self.resource = ResourceAgent(session)

    def run_orchestration(self):
        print("--- ðŸ“‹ PLANNING AGENT START ---")
        
        # 1. ANALYTICS PHASE (Feature 4)
        for topic, data in KNOWLEDGE_BASE.items():
            weightage = self.analytics.analyze_pyq_weightage(data)
            self.session.topic_weightage[topic] = weightage
            print(f"[ANALYTICS] '{topic}' Weightage: {weightage:.2f}")

        # Choose the highest priority topic (high weightage AND low mastery)
        priority_topic = max(
            KNOWLEDGE_BASE.keys(), 
            key=lambda t: self.session.topic_weightage.get(t, 0) * (1 - self.session.performance_history.get(t, 0))
        )
        self.session.current_topic = priority_topic
        print(f"\nPLAN: Prioritizing '{priority_topic}' for study.")

        # 2. RESOURCE INGESTION PHASE (Feature 5)
        self.resource.generate_exam_notes(self.session, KNOWLEDGE_BASE[priority_topic]["raw_text"])
        
        # 3. LEARNING PHASE (Multi-Agent System & Socratic Tutor)
        tutor_agent = SocraticTutorAgent(self.session)
        status = ""
        for i, response in enumerate(SIMULATED_USER_RESPONSES[:2]): # Use first 2 answers for Socratic loop
            status = tutor_agent.run_socratic_loop(response)
            if status == "COMPLETED":
                break

        # 4. ASSESSMENT PHASE (Practice Test)
        if status == "COMPLETED" or tutor_agent.turn >= 2:
            print("\nMoving to Assessment...")
            assessment_agent = AssessmentAgent(self.session)
            assessment_agent.run_assessment()

        print(f"\nFINAL MASTERY STATUS: {self.session.performance_history}")
        print("--- âœ… ORCHESTRATION END ---")


# --- Execution ---
if __name__ == "__main__":
    topics = list(KNOWLEDGE_BASE.keys())
    user_session = SessionData(user_name="Aspirant", topic_list=topics)
    
    planner = PlanningAgent(user_session)
    planner.run_orchestration()

