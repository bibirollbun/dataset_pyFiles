import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


# Research Agent: Its job is to use the google_search tool and present findings.
research_agent = LlmAgent(
    name="ResearchAgent",
    model=Gemini(model="gemini-2.5-flash-lite",retry_option=retry_config),
    instruction="""You are an expert on Massachusetts public utilities. Your only job is to use the
    google_search tool to find 3-4 pieces of relevant information on the topic of 
    Massachusetts Municipal Light Plants (MLPs) for the time period specified in the prompt.
    Present the findings. Findings MUST HAVE citations and url links to the information.""",
    tools=[google_search],
    output_key="research_findings", # The result of this agent will be stored in the session state with this key.
)

print("âœ… research_agent created.")


# Outline Agent: Creates the initial newsletter outline.
outline_agent = Agent(
    name="OutlineAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Create a newsletter outline using {research_findings} with:
    1. A catchy headline
    2. An introduction hook
    3. 3-4 sections with subheadings
    4. Include citations and url links to the information
    4. A thank you for reading ending""",
    output_key="newsletter_outline", # The result of this agent will be stored in the session state with this key.
)

print("âœ… outline_agent created.")


# Writer Agent: Writes the full blog post based on the outline from the previous agent.
writer_agent = Agent(
    name="WriterAgent",
    model="gemini-2.5-flash-lite",
    # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
    instruction="""Following this outline strictly: {newsletter_outline}
    Write a brief newsletter of strictly no more than 800 words with an 
    engaging and informative tone.""",
    output_key="newsletter_draft", # The result of this agent will be stored with this key.
)

print("âœ… writer_agent created.")


# Editor Agent: Edits and polishes the draft from the writer agent.
editor_agent = Agent(
    name="EditorAgent",
    model="gemini-2.5-flash-lite",
    # This agent receives the `{newsletter_draft}` from the writer agent's output.
    instruction="""Edit this draft: {newsletter_draft}
    Your task is to polish the text by fixing any grammatical errors, 
    improving the flow and sentence structure, and enhancing overall clarity. 
    Make sure the word count does not exceed 800 words.""",
    output_key="final_newsletter", # This is the final output of the entire pipeline.
)

print("âœ… editor_agent created.")


root_agent = SequentialAgent(
    name="NewsletterPipeline",
    sub_agents=[research_agent,outline_agent, writer_agent, editor_agent],
)

print("âœ… Sequential Agent created.")


runner = InMemoryRunner(agent=root_agent)
response = await runner.run_debug("Write a newsletter covering news during October 2025")
# print(response.text)


print(response)

