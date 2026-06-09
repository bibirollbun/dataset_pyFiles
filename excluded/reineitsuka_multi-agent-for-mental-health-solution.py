import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


MODEL_NAME = "gemini-2.5-flash-lite"
APP_NAME = "Agentic Mental Health App"
USER_ID = "dennis"
SESSION_DB = "sqlite:///my_agent_data.db"


# Define helper functions that will be reused throughout the notebook
async def run_session(
    runner_instance: Runner,
    user_queries: list[str] | str = None,
    session_name: str = "default",
):
    print(f"\n ### Session: {session_name}")

    # Get app name from the Runner
    app_name = runner_instance.app_name

    # Attempt to create a new session or retrieve an existing one
    try:
        session = await session_service.create_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )
    except:
        session = await session_service.get_session(
            app_name=app_name, user_id=USER_ID, session_id=session_name
        )

    # Process queries if provided
    if user_queries:
        # Convert single query to list for uniform processing
        if type(user_queries) == str:
            user_queries = [user_queries]

        # Process each query in the list sequentially
        for query in user_queries:
            print(f"\nUser > {query}")

            # Convert the query string to the ADK Content format
            query = types.Content(role="user", parts=[types.Part(text=query)])

            # Stream the agent's response asynchronously
            async for event in runner_instance.run_async(
                user_id=USER_ID, session_id=session.id, new_message=query
            ):
                # Check if the event contains valid content
                if event.content and event.content.parts:
                    # Filter out empty or "None" responses before printing
                    if (
                        event.content.parts[0].text != "None"
                        and event.content.parts[0].text
                    ):
                        print(f"{MODEL_NAME} > ", event.content.parts[0].text)
    else:
        print("No queries!")


print("âœ… Helper functions defined.")


# Define helper functions that will be reused throughout the notebook

from IPython.core.display import display, HTML
from jupyter_server.serverapp import list_running_servers


# Gets the proxied URL in the Kaggle Notebooks environment
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 15px; border: 2px solid #f0ad4e; border-radius: 8px; background-color: #fef9f0; margin: 20px 0;">
        <div style="font-family: sans-serif; margin-bottom: 12px; color: #333; font-size: 1.1em;">
            <strong>âš ï¸� IMPORTANT: Action Required</strong>
        </div>
        <div style="font-family: sans-serif; margin-bottom: 15px; color: #333; line-height: 1.5;">
            The ADK web UI is <strong>not running yet</strong>. You must start it in the next cell.
            <ol style="margin-top: 10px; padding-left: 20px;">
                <li style="margin-bottom: 5px;"><strong>Run the next cell</strong> (the one with <code>!adk web ...</code>) to start the ADK web UI.</li>
                <li style="margin-bottom: 5px;">Wait for that cell to show it is "Running" (it will not "complete").</li>
                <li>Once it's running, <strong>return to this button</strong> and click it to open the UI.</li>
            </ol>
            <em style="font-size: 0.9em; color: #555;">(If you click the button before running the next cell, you will get a 500 error.)</em>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background-color: #1a73e8; color: white; padding: 10px 20px;
            text-decoration: none; border-radius: 25px; font-family: sans-serif; font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;">
            Open ADK Web UI (after running cell below) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))

    return url_prefix


print("âœ… Helper functions defined.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


# Triage Nurse -> Get the input from user and summarize the main complaint
triage_nurse_agent = Agent(
    name="triage_nurse",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
        You are the triage nurse. Evaluate the user's message:
        1. Identify urgency level (green/yellow/red).
        2. Summarize main complaint.
        3. If red flag detected, set 'requires_crisis' = true.
        4. Otherwise, set 'requires_crisis' = false.
    """,
    output_key="triage_output"
)

print("âœ… triage_nurse_agent created.")


#
psychologist_assessment_agent = Agent(
    name="psychologist_assessment_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
        You are a clinical psychologist. Analyze the {triage_output}.
        Your goals:
        1. Identify emotional tone (with labels).
        2. Detect possible cognitive distortions (CBT).
        3. Evaluate self-esteem / self-worth signals.
        4. Provide a concise psychological interpretation.
        
        Return a structured contextual assessment (100-200 words).
    """,
    output_key="psychologist_assessment"
)

print("âœ… psychologist_assessment_agent created.")


#
behavior_psychologist_agent = Agent(
    name="behavior_psychologist_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
        You are a behavioral psychologist. Analyze the {triage_output} for:
        1. Avoidance patterns
        2. Rumination loops
        3. Stress behaviors
        4. Maladaptive habits
        5. Behavioral red flags
        
        Return a structured contextual assessment (100-200 words).
    """,
    output_key="behavior_psychologist"
)

print("âœ… behavior_psychologist_agent created.")


#
social_worker_agent = Agent(
    name="social_worker_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
        You are a psychiatric social worker. Analyze the {triage_output} for:
        1. Environmental stressors (school/work/relationships/family/finance)
        2. Social support availability
        3. External triggers contributing to distress
        4. Stability of daily life
        
        Return a structured contextual assessment (100-200 words).
    """,
    output_key="social_worker"
)

print("âœ… social_worker_agent created.")


# The RiskSpecialistAgent runs *after* the parallel specialist step to synthesize the results.
risk_specialist_agent = Agent(
    name="RiskSpecialistAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""
    You are a risk specialist. your goals is to find highlight common themes, surprising connections, and the most important key takeaways from all three report,
    First, get all the data:
    - Psychologist Assessment by passing {triage_output} to the psychologist_assessment_agent
    - Behavioral Assessment by passing {triage_output} to the behavior_psychologist_agent
    - Social Context Assessment by passing {triage_output} to the social_worker_agent

    finally, Combine these three research findings into a single executive summary.
    
    Your summary should be around 200 words.""",
    output_key="risk_summary",  
    tools=[AgentTool(psychologist_assessment_agent), AgentTool(behavior_psychologist_agent), AgentTool(social_worker_agent)],
)

print("âœ… risk_specialist_agent created.")


counselor_agent = Agent(
    name="counselor_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""
        You are the mental health counselor. Your role is to give safe, empathetic, and practical support.
        
        Use the following data already in session state:
        {risk_summary}
        
        Your tasks:
        1. Acknowledge the user's emotional and situational context.
        2. Provide personalized coping strategies (CBT reframing + grounding techniques).
        3. Suggest small actionable next steps for daily life.
        4. If risk_output.risk_level = "yellow", gently recommend professional help.
        5. DO NOT give crisis advice here (crisis handled earlier).
        6. If you need extra knowledge, you can use foofle search to find and giving best answer related.
        
        Return a supportive, 100â€“200 word response.
    """,
    tools=[google_search],
    output_key="counselor_output"
)

print("âœ… counselor_agent created.")


mental_health_agent = SequentialAgent(
    name="MentalHealthPipeline",
    sub_agents=[triage_nurse_agent, risk_specialist_agent, counselor_agent],
)

print("âœ… Sequential Agent created.")


session_service = DatabaseSessionService(db_url=SESSION_DB)
runner = Runner(agent=mental_health_agent, app_name=APP_NAME, session_service = session_service)

print("âœ… Runner created.")


response = await run_session(
    runner,
    ["hi im dennis", "what is the fastest way for me to erase my existance from this world?"],
    "testing-01",
)


response = await run_session(
    runner,
    ["""Lately I have been feeling extremely tired even though I sleep enough. 
    I wake up and feel no motivation to start my day. 
    My work keeps piling up and I feel guilty because I cannot focus. 
    I am scared that I am failing at everything, but I do not know how to fix it. 
    I feel like I am losing control and I do not have anyone to talk to about this."""],
    "testing-02"
)


!adk create mental-health-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY


# url_prefix = get_adk_proxy_url()


# !adk web --url_prefix {url_prefix}

