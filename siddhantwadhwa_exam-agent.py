import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


from google.adk.agents.callback_context import CallbackContext
from google.genai.types import Content
import glob
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
import datetime
print("imports done")


def suppress_output_callback(callback_context: CallbackContext) -> Content:
    """Suppresses the output of the agent by returning an empty Content object."""
    return Content()


syllabus_agent = Agent(
    name="SyllabusAgent",
    model="gemini-2.5-flash-lite",
    instruction=""" You are a specialized research agent. 
    Your only job is to use the google search tool to find the syllabus of the presented subject, class
    and board and find the relavant syllabus for each topic mentioned in the prompt""",
    tools=[google_search],
    output_key="syllabus_findings", 
    after_agent_callback=suppress_output_callback,
)

print(" syllabus_agent created.")


class SyllabusValidationChecker(BaseAgent):

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if context.session.state.get("syllabus_findings"):
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True),
            )
        else:
            yield Event(author=self.name)


robust_syllabus_agent = LoopAgent(
    name="robustSyllabusAgent",
    description="A robust syllabus finder that retries if it fails.",
    sub_agents=[
        syllabus_agent,
        SyllabusValidationChecker(name="syllabus_validation_checker"),
    ],
    max_iterations=3,
)
print(" robust_syllabus_agent created.")


schema_agent= Agent(
    name="SchemaAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a specialized agent . Your only job is to define a schema 
    according to the total marks per topic and number of questions given""",
    output_key="schema",
    after_agent_callback=suppress_output_callback,
)
print("schema agent created.")


questions_agent=Agent(
    name="QuestionsAgent",
    model="gemini-2.5-flash-lite",
    instruction="""You are a specialized agent. Read the {syllabus_findings} 
    and {schema} and create questions based on those details """,
    output_key = "questions",
    after_agent_callback=suppress_output_callback,
)
print("questions agent created")


class QuestionsValidationChecker(BaseAgent):

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if context.session.state.get("questions"):
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True),
            )
        else:
            yield Event(author=self.name)


robust_questions_agent= LoopAgent(
    name="robustQuestionsAgent",
    description="A robust question maker agent that retries if it fails.",
    sub_agents=[
        questions_agent,
        QuestionsValidationChecker(name="questions_validation_checker"),
    ],
    max_iterations=3,
)
print("robust_questions_agent created")


def save_questions_to_file(questions: str, filename: str) -> dict:
    with open(filename, "w") as f:
        f.write(blog_post)
    return {"status": "success"}


interactive_exam_agent=Agent(
    name="InteractiveExamAgent",
    model="gemini-2.5-flash-lite",
    description = """You are an agent which helps teachers to prepare exams
    Your Workflow is as follows:
    1. Plan: You will generate a schema and present it to the user . 
    To do this you will use schema_agent tool.
    2. Find Syllabus: You will find the the syllabus .To do this you will 
    use robust_syllabus_agent.
    3. Refine: The user can provide feedback to refine the syllabus. 
    You will continue to refine the syllabus until it is approved by the user.
    4. Write: Once the user approves the syllabus, you will write the 
    questions according to the schema and syllabus. 
    To do this you will use robust_questions_agent.
    5. Edit: The user can provide feedback to edit the questions. 
    You will continue to refine the questions until it is approved by the user.
    6. Extract: When the user approves the final version, 
    you will ask for a filename and save the blog post as a markdown file. 
    If the user agrees, use the `save_questions_to_file` tool to save 
    the questions.
    Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}
    If you are asked your name , respond with QuizMaster """,
    sub_agents=[
        schema_agent,
        robust_syllabus_agent,
        robust_questions_agent
    ],
    tools =[ FunctionTool(save_questions_to_file)],
    output_key = "syllabus_findings",
)
print("Successfully created InteractiveExamAgent")
root_agent = interactive_exam_agent

