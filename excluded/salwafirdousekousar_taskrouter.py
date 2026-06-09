!pip install -q google-generativeai
!pip install -q rich
google-api-python-client


import google.generativeai as genai
from rich import print
import time
import json
GEMINI_API_KEY=input("enter key:")
genai.configure(api_key=GEMINI_API_KEY)


def search_tool(query: str):
    return f"[search results for '{query}']:This is a simulated response used only for demonstration."


class BaseAgent:
    def __init__(self,name):
        self.name=name
    def run(self,prompt,context):
        raise NotImplementedError


class PlannerAgent(BaseAgent):
    def run(self, prompt, context):
        system = (
            "You are a planning agent. Break the user's task into clear actionable steps."
        )
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            [{"role": "system", "text": system},
             {"role": "user", "text": prompt}]
        )
        steps = response.text
        context["plan"] = steps
        return steps



class ResearchAgent(BaseAgent):
    def run(self, prompt, context):
        query = f"{prompt} background information"
        search_data = search_tool(query)
        context["research"] = search_data
        return search_data



class WriterAgent(BaseAgent):
    def run(self, prompt, context):
        system = (
            "You are a writing/solution agent. Use the plan and research to create the final polished output."
        )
        final_prompt = f"""
Task: {prompt}

Plan:
{context.get('plan')}

Research:
{context.get('research')}
"""
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            [{"role": "system", "text": system},
             {"role": "user", "text": final_prompt}]
        )
        final_output = response.text
        context["final"] = final_output
        return final_output



class TaskRouter:
    def __init__(self):
        self.agents = [
            PlannerAgent("Planner"),
            ResearchAgent("Researcher"),
            WriterAgent("Writer")
        ]
        self.context = {}

    def run(self, prompt):
        print("[bold blue]Starting multi-agent pipeline...[/bold blue]")
        start_time = time.time()

        for agent in self.agents:
            print(f"\n[green]Running {agent.name} Agent...[/green]")
            output = agent.run(prompt, self.context)
            print(output)

        print("\n[bold yellow]Final Output Ready[/bold yellow]")
        total_time = round(time.time() - start_time, 2)
        print(f"[italic]Pipeline completed in {total_time} seconds[/italic]")
        return self.context["final"]



router = TaskRouter()

user_task = input("Enter your task: ")
result = router.run(user_task)

print("\n\n======================")
print("[bold magenta]FINAL RESULT[/bold magenta]")
print("======================\n")
print(result)


