# =====================================================================
#                E D U C A T I O N A L   A G E N T  (Single File)
# =====================================================================

import json

# ---------------------------------------------------------------------
# LLM Stub (Offline) – imitates LLM behavior to keep code runnable
# ---------------------------------------------------------------------
class LLMStub:
    def generate(self, prompt: str) -> str:
        """
        Very simple rule-based generator to avoid API usage.
        Produces structured JSON output depending on the agent request.
        """

        prompt_lower = prompt.lower()

        # Understanding Agent ----------------------------------------------------
        if "understanding_agent" in prompt_lower:
            return json.dumps({
                "topic": prompt.split("Topic:")[1].split("\n")[0].strip(),
                "difficulty": "Beginner",
                "learning_goals": [
                    "Understand the basic concept",
                    "Learn important components",
                    "Be able to explain the idea simply"
                ],
                "misconceptions": []
            })

        # Curriculum Agent -------------------------------------------------------
        if "curriculum_agent" in prompt_lower:
            return json.dumps({
                "syllabus": [
                    {"step": 1, "title": "Introduction"},
                    {"step": 2, "title": "Key Components"},
                    {"step": 3, "title": "How It Works"},
                    {"step": 4, "title": "Examples"},
                    {"step": 5, "title": "Summary"},
                ]
            })

        # Content Agent ---------------------------------------------------------
        if "content_agent" in prompt_lower:
            return json.dumps({
                "content": [
                    {"step": 1, "text": "This topic is important because it explains a core idea."},
                    {"step": 2, "text": "The main components involved include essential parts."},
                    {"step": 3, "text": "It works by following a simple logical process."},
                    {"step": 4, "text": "Real life examples help to understand this better."},
                    {"step": 5, "text": "To summarize, the topic connects ideas in a meaningful way."},
                ]
            })

        # Evaluation Agent ------------------------------------------------------
        if "evaluation_agent" in prompt_lower:
            return json.dumps({
                "quiz": [
                    {"question": "What is the main idea of this topic?", "answer": "It explains a basic concept."},
                    {"question": "Name one key component discussed.", "answer": "Any major element mentioned."},
                    {"question": "Why is this topic important?", "answer": "It helps you understand a core idea."}
                ]
            })

        return json.dumps({"error": "Unknown request"})


# ---------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------
class UnderstandingAgent:
    def run(self, topic: str, user_input: str, llm):
        prompt = f"""
        understanding_agent
        Topic: {topic}
        User Input: {user_input}
        """
        return json.loads(llm.generate(prompt))


class CurriculumAgent:
    def run(self, understanding_output: dict, llm):
        prompt = f"""
        curriculum_agent
        Learning Goals: {understanding_output['learning_goals']}
        """
        return json.loads(llm.generate(prompt))


class ContentAgent:
    def run(self, curriculum_output: dict, llm):
        prompt = f"""
        content_agent
        Syllabus: {curriculum_output['syllabus']}
        """
        return json.loads(llm.generate(prompt))


class EvaluationAgent:
    def run(self, content_output: dict, llm):
        prompt = f"""
        evaluation_agent
        Content: {content_output}
        """
        return json.loads(llm.generate(prompt))


# ---------------------------------------------------------------------
# ORCHESTRATOR (Main Agent)
# ---------------------------------------------------------------------
class EducationalAgent:
    def __init__(self):
        self.llm = LLMStub()
        self.understanding = UnderstandingAgent()
        self.curriculum = CurriculumAgent()
        self.content = ContentAgent()
        self.evaluation = EvaluationAgent()

    def run(self, topic: str, user_input: str):
        # Step 1 — Understanding
        step1 = self.understanding.run(topic, user_input, self.llm)

        # Step 2 — Curriculum
        step2 = self.curriculum.run(step1, self.llm)

        # Step 3 — Learning Material
        step3 = self.content.run(step2, self.llm)

        # Step 4 — Quiz
        step4 = self.evaluation.run(step3, self.llm)

        # Final Result
        return {
            "Understanding": step1,
            "Curriculum": step2,
            "Content": step3,
            "Evaluation": step4
        }


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------
if __name__ == "__main__":
    agent = EducationalAgent()

    # You can change the topic here:
    topic = "Photosynthesis"
    user_input = "Explain this topic in a simple way."

    result = agent.run(topic, user_input)

    print("\n==================== FINAL OUTPUT ====================\n")
    print(json.dumps(result, indent=4))


