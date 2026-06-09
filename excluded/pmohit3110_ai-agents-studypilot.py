# --- Kaggle Notebook Cell 1: Setup and Imports ---

import os
import google.generativeai as genai
from collections import deque # For simple memory/context
import json # For tool output parsing

# NEW: Correct way to access secrets in Kaggle Notebooks
from kaggle_secrets import UserSecretsClient

# Initialize the secrets client
user_secrets = UserSecretsClient()

# Get your GEMINI_API_KEY
try:
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
except Exception as e:
    raise ValueError(f"Failed to retrieve GEMINI_API_KEY from Kaggle Secrets: {e}. "
                     "Please ensure it's added and enabled in the 'Secrets' tab.")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model (using a suitable model for text generation)
# You might choose 'gemini-pro' or other available models.
GENERATION_MODEL = genai.GenerativeModel('models/gemini-2.5-flash')
print("Setup complete! Gemini model initialized.")


# --- Kaggle Notebook Cell 2: Define Tools ---

# A mock Google Search tool.
# In a real scenario, you'd integrate with a real search API (e.g., Google Custom Search API)
# For the Capstone, a well-commented mock is acceptable for demonstrating the concept.
def google_search_tool(query: str) -> str:
    """
    Simulates a Google Search. In a real application, this would call a real search API.
    Returns a string summary of search results.
    """
    print(f"DEBUG: Performing mock search for: '{query}'")
    # Mock data for demonstration purposes
    if "French Revolution" in query:
        return (
            "The French Revolution (1789-1799) was a period of radical political and societal change in France. "
            "It began with the storming of the Bastille, leading to the overthrow of the monarchy, "
            "the rise of the First French Republic, and eventually the Reign of Terror. "
            "Key figures included Louis XVI, Marie Antoinette, Maximilien Robespierre. "
            "Its causes included economic hardship, social inequality, and Enlightenment ideas. "
            "It ended with the rise of Napoleon Bonaparte."
        )
    elif "Photosynthesis" in query:
        return (
            "Photosynthesis is the process by which green plants and some other organisms "
            "use sunlight to synthesize foods with the help of chlorophyll. "
            "Raw materials are carbon dioxide and water. Products are glucose and oxygen. "
            "It occurs mainly in the chloroplasts of plant cells. "
            "Two main stages: light-dependent reactions and light-independent reactions (Calvin Cycle)."
        )
    elif "cars" in query:
         return (
            "A car (or automobile) is a wheeled motor vehicle used for transportation. "
            "Most definitions specify that cars are designed to run primarily on roads, "
            "to have seating for one to eight people, to typically have four wheels, "
            "and to be constructed principally for the transport of people rather than goods."
         )
    else:
        return f"No specific mock data for '{query}'. General search result: Information about {query} is widely available online."

# Registering the tool (if your ADK requires explicit registration)
# For simple Python scripts, you just call the function.
TOOLS = {
    "google_search": google_search_tool
}

print("Tools defined: google_search")


# --- Kaggle Notebook Cell 3: Define Agent Classes (Multi-Agent System) ---

class Agent:
    def __init__(self, name: str, description: str, model):
        self.name = name
        self.description = description
        self.model = model
        print(f"Agent '{self.name}' initialized.")

    def act(self, task: str, context: str = "") -> str:
        """Base method for an agent to perform its action."""
        raise NotImplementedError("Subclasses must implement 'act' method.")

class ResearchAgent(Agent):
    def __init__(self, model, search_tool):
        super().__init__("ResearchAgent", "Researches topics using a search tool.", model)
        self.search_tool = search_tool

    def act(self, topic: str) -> str:
        prompt = (
            f"You are the ResearchAgent. Your task is to find comprehensive information "
            f"about the topic: '{topic}'. Use the provided search tool to gather facts. "
            f"Summarize the key findings from your search in detail. "
            f"Search query: '{topic} overview' and '{topic} key facts'."
        )
        # In a more advanced system, the agent would decide its search query.
        # For simplicity, we'll use the topic directly.
        search_results = self.search_tool(topic)

        # Use Gemini to summarize the search results more effectively if needed
        response = self.model.generate_content(
            f"Based on the following raw search results about '{topic}', "
            f"provide a concise yet informative summary of the key points:\n\n"
            f"Search Results: {search_results}\n\nSummary:"
        )
        return response.text

class TeacherAgent(Agent):
    def __init__(self, model):
        super().__init__("TeacherAgent", "Creates study guides from research.", model)

    def act(self, topic: str, research_summary: str) -> str:
        prompt = (
            f"You are the TeacherAgent. Your goal is to create an easy-to-understand "
            f"study guide for the topic '{topic}', based on the provided research summary. "
            f"The guide should be in bullet points and cover key concepts, important dates/figures (if applicable), "
            f"and main ideas. Make it concise and clear for a student."
            f"\n\nResearch Summary:\n{research_summary}"
            f"\n\nStudy Guide for '{topic}':"
        )
        response = self.model.generate_content(prompt)
        return response.text

class ExaminerAgent(Agent):
    def __init__(self, model):
        super().__init__("ExaminerAgent", "Generates quizzes from study guides.", model)

    def act(self, topic: str, study_guide: str) -> str:
        prompt = (
            f"You are the ExaminerAgent. Create a 5-question multiple-choice quiz about '{topic}', "
            f"based *only* on the following study guide. For each question, provide 4 options (A, B, C, D) "
            f"and clearly indicate the correct answer (e.g., 'Correct Answer: B'). "
            f"Ensure the questions test understanding of the main points."
            f"\n\nStudy Guide:\n{study_guide}"
            f"\n\nQuiz for '{topic}':"
        )
        response = self.model.generate_content(prompt)
        return response.text

class CoordinatorAgent(Agent):
    def __init__(self, model, research_agent, teacher_agent, examiner_agent):
        super().__init__("CoordinatorAgent", "Orchestrates the workflow between other agents.", model)
        self.research_agent = research_agent
        self.teacher_agent = teacher_agent
        self.examiner_agent = examiner_agent
        self.session_memory = {"current_topic": None} # Simple memory

    def process_request(self, user_query: str) -> str:
        # Simple NLU to determine intent and topic
        if "learn about" in user_query.lower() or "study" in user_query.lower():
            # Extract topic from query (basic extraction)
            if "learn about" in user_query.lower():
                topic = user_query.split("learn about")[-1].strip().replace("?", "").strip()
            elif "study about" in user_query.lower():
                topic = user_query.split("study about")[-1].strip().replace("?", "").strip()
            else:
                topic = user_query.replace("study", "").replace("learn", "").strip().replace("?", "").strip()

            if not topic: # Fallback if extraction is too generic
                 return "Please specify a topic you want to learn about, e.g., 'Learn about Photosynthesis'."


            self.session_memory["current_topic"] = topic
            print(f"COORDINATOR: User wants to learn about '{topic}'.")

            # 1. Research Phase
            print("COORDINATOR: Activating ResearchAgent...")
            research_summary = self.research_agent.act(topic)
            print("COORDINATOR: ResearchAgent complete.")

            # 2. Teaching Phase
            print("COORDINATOR: Activating TeacherAgent...")
            study_guide = self.teacher_agent.act(topic, research_summary)
            print("COORDINATOR: TeacherAgent complete.")

            # 3. Examination Phase
            print("COORDINATOR: Activating ExaminerAgent...")
            quiz = self.examiner_agent.act(topic, study_guide)
            print("COORDINATOR: ExaminerAgent complete.")

            return (
                f"### Your StudyPilot Guide for '{topic}'\n\n"
                f"**Research Summary:**\n{research_summary}\n\n"
                f"**Study Guide:**\n{study_guide}\n\n"
                f"**Quiz Time!**\n{quiz}\n\n"
                f"I hope this helps you learn about {topic}!"
            )
        elif "quiz me again" in user_query.lower() and self.session_memory["current_topic"]:
            topic = self.session_memory["current_topic"]
            print(f"COORDINATOR: User wants another quiz on '{topic}'.")
            # For simplicity, we'll re-run the whole process to generate a new quiz
            # In a real system, you might store the guide in memory to avoid re-generating.
            research_summary = self.research_agent.act(topic)
            study_guide = self.teacher_agent.act(topic, research_summary)
            quiz = self.examiner_agent.act(topic, study_guide)
            return (
                f"### Here's another quiz on '{topic}'!\n\n"
                f"**Quiz Time!**\n{quiz}\n\n"
                f"Good luck!"
            )
        else:
            return "I can help you learn about a topic! Just tell me, e.g., 'Learn about Photosynthesis'."
print("Agent classes defined.")


# --- Kaggle Notebook Cell 4: Main Execution Block ---

# Initialize the agents
research_agent_instance = ResearchAgent(model=GENERATION_MODEL, search_tool=TOOLS["google_search"])
teacher_agent_instance = TeacherAgent(model=GENERATION_MODEL)
examiner_agent_instance = ExaminerAgent(model=GENERATION_MODEL)

# Initialize the Coordinator (our main entry point)
studypilot_coordinator = CoordinatorAgent(
    model=GENERATION_MODEL, # Coordinator might also use Gemini for NLU
    research_agent=research_agent_instance,
    teacher_agent=teacher_agent_instance,
    examiner_agent=examiner_agent_instance
)

print("\n--- StudyPilot is ready! ---")
print("You can start by asking, e.g., 'Learn about The French Revolution'")
print("Or 'Study about Photosynthesis'\n")


# Example interactions:
user_input_1 = "Learn about Photosynthesis"
response_1 = studypilot_coordinator.process_request(user_input_1)
print(response_1)

print("\n---------------------------------------------------\n")

user_input_2 = "Study about The French Revolution"
response_2 = studypilot_coordinator.process_request(user_input_2)
print(response_2)

print("\n---------------------------------------------------\n")

user_input_3 = "Quiz me again!" # Testing memory
response_3 = studypilot_coordinator.process_request(user_input_3)
print(response_3)

print("\n---------------------------------------------------\n")

user_input_4 = "Tell me about cars" # Topic not in mock search, but generic works
response_4 = studypilot_coordinator.process_request(user_input_4)
print(response_4)




