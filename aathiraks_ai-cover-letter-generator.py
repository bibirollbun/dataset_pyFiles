!pip install pdfplumber
!pip install -q trafilatura


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner, InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.plugins.logging_plugin import (
    LoggingPlugin,
)

print("âœ… ADK components imported successfully")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


import pdfplumber
import re
def read_pdf_text(file_path: str):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    
    # Normalize Whitespace and Line Breaks
    cleaned_text = re.sub(r'[\r\n\t]+', ' ', text)

    # Collapse multiple spaces
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

    # Handle bullet points
    cleaned_text = re.sub(r'[â€¢\-â€“â€”*]+', ' ', cleaned_text)

    # Remove non-ASCII or control characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)

    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return {
        "status": "success",
        "data": cleaned_text
    }
    
read_resume_tool = FunctionTool(read_pdf_text)


json_schema_request = {
    "skills": ["list of key technical skills"],
    "experience": [
        {
            "title": "Job Title",
            "company": "Company Name",
            "duration": "Duration (e.g. 2020 - 2023)",
            "description": "a summary of responsibilities"
        }
    ],
    "education": [
        {
            "degree": "Degree/Certification Name",
            "institution": "University/Institution Name",
            "year": "Graduation Year",
            "description": "description, if any. else keep it empty"
        }
    ],
    "certification": ["list of certifications if any"],
    "summary": "summary provided by the candidate"
}


# This agent runs ONCE at the beginning to create the first draft.
resume_reader = Agent(
    name="ResumeReader",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=f"""You are a specialized AI designed ONLY to perform data extraction. 
    Your sole task is to complete the following mechanical steps without any conversation, explanation, or confirmation:

    1. You MUST call the `read_pdf_text()` tool. Identify the file path in the user's message and use it as the argument for the tool.
    2. Extract data based on this target JSON schema: {json_schema_request}.

    Always return the output as a single, valid JSON object, and nothing else. 
    If a field is missing, use an empty string or an empty list as the value.

    """,    
    tools=[read_resume_tool],
    output_key="resume", #Stores the first draft in the state.
)

print("âœ… resume_reader created.")


import trafilatura
def read_job_description(url: str) -> str:
    """
    Download a job posting page and extract its main textual content. Works best on standard job boards/company career pages.
    """

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {
            "status": "error",
            "error_message": "Job description not downloaded"
        }

    text = trafilatura.extract(downloaded)

    if text:
        return {
            "status": "success",
            "job_desc": text
        }

    else:
        return {
            "status": "error",
            "error_message": "Job description not available"
        }
read_jd_tool = FunctionTool(read_job_description)


# This agent runs ONCE at the beginning to create the first draft.
initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction=f"""You are an expert career coach that writes concise, tailored cover letters. 
    Here are the instructions:
    1. The parsed resume (JSON string) is available in your context under the key `resume`. Analyze it.
    2. Use `read_job_description()` to read the job description.
    3. Check the "status" field in each tool's response for errors. 
    4. Based on the job description and the user's resume, write a cover letter. Keep it to about one page, with: introduction, 1-2 body parapraphs, and a closing. 
    5. If the user provides extra instructions, use them. 
    6. If a required field is missing from the resume JSON or job description, do NOT invent it; instead, omit it or describe it at a higher level without fabricating specifics. 
    7. Explicitly connect at least 2â€“3 concrete experiences from the resume to concrete responsibilities or methods mentioned in the job description 
    Use a professional, confident tone. Emphasize the most relevant experience, skills, and measurable impact.
    Do NOT invent new experience or skills beyond the resume. 
    Output only the cover letter, with no introduction or explanation.""",
    tools=[read_jd_tool],
    output_key="cover_letter", #Stores the first draft in the state.
)

print("âœ… initial_writer_agent created.")


# This agent's only job is to provide feedback or the approval signal. It has no tools.
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a hyper-critical cover letter reviewer and judge. Your job is to ensure the letter is publication-ready and human-written.

    Review the cover letter provided below, focusing specifically on:
    1. Redundancy & Flow: Are any sentences, phrases, or project descriptions repeated (e.g., repeating the Master's thesis)? Does the argument flow logically without jumping?
    2. LLM Jargon: Does the tone sound overly generic, formal, or use common LLM filler phrases (e.g., "culminating in," "immense enthusiasm," "keen interest," "groundbreaking research")?
    3. Alignment & Impact: Does it clearly link the user's specific skills (e.g., Anomaly Detection) directly to the specific job requirements (e.g., Data mining for diagnostics) without ambiguity?
    4. Overclaiming / Hallucination: Flag any skills, tools, or achievements not explicitly present in the resume JSON as issues.
    Cover letter: {cover_letter}

    Evaluate the cover letter based on the three criteria above.
    - If the cover letter passes all three criteria perfectly (no repetition, human tone, strong alignment), you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement, starting with the most critical flaw (e.g., "The section on the Master's thesis is repeated in paragraph two and three; condense the description.").""",
    output_key="critique", #Stores the feedback in the state
)

print("âœ… critic_agent created.")


# # Saves the cover letter to a file
# import datetime
# def save_cover_letter(cover_letter):
#     time = datetime.datetime().now().strftime("%Y-%m-%d")
#     with open(f"CoverLetter_{time}", "w") as f:
#         f.write(cover_letter)
#     return {"status": "success"}


# This is the function that the RefinerAgent will call to exit the loop.

def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the cover letter is finished and no more changes are needed."""
    return {"status": "approved", "message": "Cover letter approved. Exiting refinement loop."}

print("âœ… exit_loop function created.")

exit_loop_func = FunctionTool(exit_loop)


def human_critique_tool(draft):
    """Long-Running Operation: Simulates sending the final draft to a human reviewer. 
    This function pauses the agent's execution until an external signal (Human's critique or final APPROVAL) is received."""
    print(f"\n[--- Human-in-the-Loop Triggered: Cover letter sent for FINAL review. ---")
    print(f"Cover Letter: \n {draft}")
    print("Agent Paused. External logic must provide 'APPROVED' or a critique to resume")

    return {"status": "PAUSED_FOR_HUMAN_REVIEW", "draft": draft}

human_critique_tool_func = FunctionTool(human_critique_tool)
print("âœ… Human critique tool function created.")


# This agent refines the cover letter based on critique OR calls the exit_loop function.
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a cover letter refiner. You have a cover letter draft and critique.
    
    Cover letter Draft: {cover_letter}
    Critique: {critique}

    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and return the existing cover letter draft unchanged.
    - OTHERWISE, rewrite the cover letter draft to fully incorporate the feedback from the critique.""",
    output_key="cover_letter", #It overwrites the cover letter with the new, refined version.
    tools=[
        FunctionTool(exit_loop)
    ], #The tool is now correctly initialized with the function reference.
)

print("âœ… refiner_agent created. ")


# The LoopAgent contains the agents that will run repeatedly: Critic -> Refiner.
cover_letter_refinement_loop = LoopAgent(
    name="CoverLetterRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=5, # Prevents infinite loop
)

# The root agent is a sequential agent that defines the overall work flow: Initial write -> Refinement Loop.
root_agent = SequentialAgent(
    name="Pipeline",
    sub_agents=[resume_reader, initial_writer_agent, cover_letter_refinement_loop],
)

print("âœ… Loop and Sequential Agents created.")


runner = InMemoryRunner(agent=root_agent, 
                       plugins=[LoggingPlugin()])


pdf_path = "//kaggle/input/sampleresume/SampleResume_Canva.pdf"
url = "https://careers.pvh.com/jobs/senior-business-analyst-analytics-amsterdam-noord-holland-netherlands?source=LinkedIn&utm_source=LinkedIn"

response = await runner.run_debug(
    f"""Write a cover letter based on the job description and resume text. 
    Resume: {pdf_path}, Job description: {url}
    """
    
)




