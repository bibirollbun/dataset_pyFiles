!pip install gradio


import gradio as gr

class RoboFlowAgent:
    def summarize(self, text: str) -> str:
        lines = text.split(".")
        summary = ". ".join(lines[:2])
        return f"Summary → {summary.strip()}."

    def meal_plan(self, calories: int = 1800):
        return {
            "target_calories": calories,
            "breakfast": "Oats + peanut butter + banana",
            "lunch": "Dal + rice + vegetable sabzi",
            "dinner": "Roti + paneer + salad",
            "snack": "Dry fruits or fruit bowl"
        }

    def study_plan(self, hours: int = 2):
        return {
            "total_hours": hours,
            "plan": [
                {"task": "Concept learning", "time": f"{hours*0.5:.1f} hr"},
                {"task": "Practice questions", "time": f"{hours*0.3:.1f} hr"},
                {"task": "Revision", "time": f"{hours*0.2:.1f} hr"},
            ]
        }

    def chat(self, text: str, mode: str):
        if mode == "Summarize":
            return self.summarize(text)
        elif mode == "Meal Plan":
            return str(self.meal_plan())
        elif mode == "Study Plan":
            return str(self.study_plan())
        else:
            return "Invalid mode selected."

agent = RoboFlowAgent()


def process(text, mode):
    return agent.chat(text, mode)

ui = gr.Interface(
    fn=process,
    inputs=[
        gr.Textbox(label="Enter your text"),
        gr.Radio(["Summarize", "Meal Plan", "Study Plan"], label="Choose Action")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RoboFlow AI Agent",
    description="A simple multi-function AI assistant for summaries, meal plans, and study scheduling."
)

ui.launch()

