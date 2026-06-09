#  Study Assistant - Multi AgentStudy Assistant - Multi Agent System Notebook


# using two agents:
#  1. QuestionAnswerAgent (LLM-powered)
#  2. EvaluationAgent (checks correctness & provides feedback)


# Install required packages if running in Colab or local Jupyter
# !pip install openai

from dataclasses import dataclass

# -------------------------------------------------------------
# Agent 1: Question Answering Agent
# -------------------------------------------------------------
@dataclass
class QuestionAnswerAgent:
    name: str = "Study Assistant Agent"

    def answer(self, question: str) -> str:
        # Placeholder LLM response logic
        # In real use, connect to an LLM (OpenAI, Gemini, Llama, etc.)
        return f"This is a generated study answer for: {question}"

# -------------------------------------------------------------
# Agent 2: Evaluation Agent
# -------------------------------------------------------------
@dataclass
class EvaluationAgent:
    name: str = "Evaluation Agent"

    def evaluate(self, response: str) -> str:
        if len(response) < 10:
            return "Response is too short. Needs improvement."
        else:
            return "Response looks meaningful and structured."

# -------------------------------------------------------------
# Multi-Agent Communication Workflow
# -------------------------------------------------------------
def multi_agent_study_system(question: str):
    qa_agent = QuestionAnswerAgent()
    eval_agent = EvaluationAgent()

    generated_answer = qa_agent.answer(question)
    evaluation = eval_agent.evaluate(generated_answer)

    return generated_answer, evaluation

# -------------------------------------------------------------
# Test Example
# -------------------------------------------------------------
question = "Explain Object Oriented Programming in Python"
answer, feedback = multi_agent_study_system(question)

print("Question:", question)
print("\nGenerated Answer: ", answer)
print("\nEvaluation Feedback: ", feedback)

# -------------------------------------------------------------
# Interactive Section for User Inputs
# -------------------------------------------------------------
while True:
    user_question = input("\nEnter a question to study (or type 'exit' to stop): ")
    if user_question.lower() == "exit":
        print("\nThank you for using the Study Assistant! Goodbye.")
        break
    ans, review = multi_agent_study_system(user_question)
    print("\nAnswer: ", ans)
    print("Feedback: ", review)
# This for easy to understand Multi Agent system 


