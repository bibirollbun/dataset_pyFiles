import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    ) 




from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# NEW AGENT: Add this cell to your notebook
parser_agent_1 = LlmAgent(
    name="ParserAgent",
    description="Parses the user's raw input into structured components.",
    model=Gemini(model="gemini-2.5-flash", temperature=0, retry_options=retry_config),

    instruction="""
    You are a silent text-parsing agent.
    Do NOT add any text before or after your required output.
    Do NOT use markdown fences.

    Read the last user message from the context.
    Parse it into prompt_text: The specific essay question, including any constraints, nothing else.
      
    """,

    # â¬‡â¬‡â¬‡ IMPORTANT: return separate output keys instead of a single dict
    output_key="prompt_text",   # ADK will merge returned dict directly into session state
)

print("âœ… parser_agent created.")



# NEW AGENT: Add this cell to your notebook
parser_agent_2 = LlmAgent(
    name="ParserAgent",
    description="Parses the user's raw input into structured components.",
    model=Gemini(model="gemini-2.5-flash", temperature=0, retry_options=retry_config),

    instruction="""
    You are a silent text-parsing agent.
    Do NOT add any text before or after your required output.
    Do NOT use markdown fences.

    Read the last user message from the context.
    Parse it into user_info: The user's name and achievements, nothing else.

    """,

    # â¬‡â¬‡â¬‡ IMPORTANT: return separate output keys instead of a single dict
    output_key="user_info",   # ADK will merge returned dict directly into session state
)

print("âœ… parser_agent created.")



research_agent = LlmAgent(
    name="ResearchAgent",
    description="Researches all information regarding what are the requirements, what is the preffered structure etc for the given college's essay. using google search(store in callback requred)",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent in collect information about college enterance essays. Your only job is to use the
    google_search tool to find 4-5 pieces of relevant information(example: what are the requirements, preffered structure etc) on the given essay topic: {prompt_text} for the given college (if not found for that perticular one, use general findings for the essay topic) and present the findings with citations(dont say anything else, just tell the findings). 
    This job is to understand the requirements for the given college acting as the foundation for our essay
    just give the research findings nothing else like constrains, issues etc.""",
    tools=[google_search],
    output_key="requred",  # The result of this agent will be stored in the session state with this key.
)

print("âœ… research_agent created.")


# Summarizer Agent: Its job is to summarize the text it receives.
summarizer_agent = Agent(
    name="SummarizerAgent",
    description="Finalises what to include, remove from person's (information, achievements, etc,) according to data from research to provide bullets for content of essay; stores in content",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Use (information, achievements, etc,) of user (including all important points):{user_info}; especially for enterance requirements from the college: {requred} to collect all points for the college essay
    This job is to collect all information acting as the foundation for our essay, just provide content nothing else""",
    output_key="content",
)

print("âœ… summarizer_agent created.")


# Summarizer Agent: Its job is to summarize the text it receives.
Organiser_agent = Agent(
    name="OrganiserAgent",
    description="Organise all content recieved from Summarize agent to the format required in essay's application; stores in organ",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    # The instruction is modified to request a bulleted list for a clear output format.
    instruction="""Use information from : {content}, and organise it in required structure for a college enterance essay noted from : {requred}
    This job is to organise all information acting as the foundation for our essay, just return the organised stuff nothing else""",
    output_key="organ",
)

print("âœ… organiser_agent created.")


writer_agent = Agent(
    name="WriterAgent",
    description="Provides an initial draft for the college essay, uses word limit(given by user) and organised content from organ; stores in blog_draft",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    # The `{blog_outline}` placeholder automatically injects the state value from the previous agent's output.
    instruction="""You are the best college enterance essay writer on this planet
    Following this outline strictly and include all important topics: {organ}
    Write an initial draft for a college's entrance essay for the given topic in the given word limit, with anauthentic, conversational, and personal tone. If you need more information about user or requirements of essay, get it from {content} and {requred} respectively
    This job is to provide an initial draft acting as the main structure for our essay, JUST RETURN THE BLOG DRAFT, NOTHING ELSE""",
    output_key="blog_draft",  # The result of this agent will be stored with this key.
)

print("âœ… writer_agent created.")


Evaluater_agent = Agent(
    name="EvaluaterAgent",
    description="Acts as a college enterance essay evaluater which provides where an essay lacks, how to improve it(taking information from requred and blog_draft); stores in Issues ",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    # This agent receives the `{blog_draft}` from the writer agent's output.
    instruction="""Command: Act as a brutal honest college enterance essay evaluater with metrics as in : {requred}, analyse the following draft: {blog_draft} and tell (where it lacks, how to improve it) 
    or approve it[(only if it would be done by someone from the comitee) by specifically saying 'APPROVED' and nothing else before nor after it, just the single word. you MUST respond with the exact phrase: "APPROVED"]
     This job is to provide problems initial draft providing solution to fix problems which may occur with main structure for our essay, JUST PROVIDE THE NEEDED OUTPUT NOTHING ELSE""",
    output_key="Issues",  # This is the final output of the entire pipeline.
)

print("âœ… Evaluater_agent.")


def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}


print("âœ… exit_loop function created.")


editor_agent = Agent(
    name="EditorAgent",
     description="Acts as best college essay refiner from issues provided in Issue as given by evaluater; stores in final_blog",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    # This agent receives the `{blog_draft}` from the writer agent's output.
    instruction="""Act as the best college enterance essay writer for topics in : {requred}, initially analyse and understand the previous draft: {blog_draft}, things to add:{content}, requirements: {requred}, Users achievements: {user_info}.
    then follow the improvment characteristics given by the essay evaluater : {Issues} and implement them in the draft. Just provide the draft nothing else
    Your job is to fix the issues in draft(our main structure) and refine it
    - IF the output by essay evaluater is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.

    """,
    output_key="blog_draft",
    tools=[
        FunctionTool(exit_loop)
    ],# This is the final output of the entire pipeline.
)

print("âœ… editor_agent created.")


Revisor_Agent = Agent(
    name="RevisorAgent",
    description="Acts as final boss to find inconsistencies from content to blog_draft (caused due to llms halucinating), and fixed them; stores in Final_essay",
    model=Gemini(
        model="gemini-2.5-pro",
        retry_options=retry_config
    ),
    instruction="""You are the final boss and act as final manual checker for writing college enterance essay, Analyse users qualities and colleges requirementf from : {content}, the read the draft: {blog_draft},
    to make sure that the previous LLMS haven't halucinated. if yes, fix that inaccurate content only to the accurate one from : {content} or users information: {user_info}, and basic facts.
    This job is to fix discrepancies which the previous things have missed. Just give the final essay and nothing else like your intro, what things are you doing etc. JUST THE FINAL ESSAY WHICH IS TO BE SUBMITTED.
    """,
output_key="Final_essay",
)
print("âœ… Revisor_Agent created")


Editing_agent = LoopAgent(
    name="EditingPipline",
    sub_agents=[Evaluater_agent, editor_agent],
    max_iterations=3
)


root_agent = SequentialAgent(
    name="WriterPipeline",
    sub_agents=[parser_agent_1,parser_agent_2,research_agent,summarizer_agent,Organiser_agent,writer_agent, Editing_agent,Revisor_Agent],
)


runner = InMemoryRunner(agent=root_agent)
Prompt= input("Provide the essay topic and your achievements: ")
response = await runner.run_debug(Prompt)







