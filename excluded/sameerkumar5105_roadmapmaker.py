
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("gemini")





import os
from kaggle_secrets import UserSecretsClient


try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("gemini")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error{e}"
    )
   



from google.adk.agents import Agent, SequentialAgent

from google.adk.models.google_llm import Gemini

from google.adk.runners import InMemoryRunner

from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService





retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], 
)


memory_service = (
    InMemoryMemoryService()
)
session_service=InMemorySessionService()



agent1=Agent(
    name="Agent1",
    # Clear, functional name for logging and tracking.
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
        
    ),
    description="Validates and cleans user topics.",
    
    instruction="""
You are Agent 1: Topic Validator.

Your job:

1. Clean the userâ€™s topic.
   - Fix grammar
   - Make it 3â€“6 words
   - Remove slang

2. Decide if the topic is researchable.
   A topic is valid only if:
   - It is a clear subject (e.g., â€œData Visualization using Pythonâ€�)
   - It is safe (no hacking/illegal/harmful)
   - It is not too vague (â€œtell me somethingâ€�, â€œhelp meâ€�)
   - It is not personal info


Strict Rules:
- Do NOT provide research.
- Do NOT rewrite instructions.
- Do NOT add extra text.
Output MUST be a single JSON object with the following keys:
- "cleaned_topic": The 3â€“6 word cleaned topic OR the original topic if invalid.
- "pipeline_action": "CONTINUE" if valid, or "STOP" if invalid/unsafe.
""",

    tools=[],
   
    output_key="validate",
    

)





agent2= Agent(
    name="Agent2",
    # Clear functional name
    model=Gemini(
        model="gemini-2.5-flash-lite",
        
        retry_options=retry_config
    ),
    description="Researches the cleaned topic.",
    instruction="""
You are Agent 2: Research Agent.

Task:
Given a cleaned topic, perform deep research using web search, websites, documentation, blogs, and trusted sources.

Rules:
1. Only include information from the last 24 months (2 years).
2. Do NOT create a roadmap.
3. Do NOT explain your process.
4. Extract and summarize only factual information.
5. Pass the information only
...
Strict Rules:
- Output MUST be a single JSON object containing all the defined research fields.
- No opinions, no roadmap, no chit chat, no pre-amble.

Definitions:
- key_concepts = fundamental ideas required to understand the topic.
- recent_trends = new developments from the last 2 years.
- important_tools = libraries, frameworks, software used for the topic.
- best_practices = updated, proven methods.
- recommended_resources = recent blogs, docs, courses (with clickable URLs).

Strict Rules:
- No opinions.
- No roadmap.
- No chit chat.
IMPORTANT SAFETY RULE:
- If the input is not valid research JSON from Agent 1, or if the topic is unsafe, harmful, illegal, or rejected by Agent 1, you MUST output exactly:
"INVALID_REQUEST"

Do NOT attempt to be helpful.
Do NOT generate educational content.
Just output: INVALID_REQUEST
""",
    
    tools=[google_search],
    output_key="resource",
)



agent3= Agent(
    name="Agent3",
    # Clear functional name.
    model=Gemini(
        model="gemini-2.5-flash-lite",
        # Uses 'flash-lite', which is ideal for this **structured text generation** # task (i.e., transforming JSON into a list format).
        retry_options=retry_config,
        session_service=session_service,
        memory_service=memory_service,
    ),
    description="Researches the cleaned topic.",
    instruction="""
You are Agent 3: Roadmap Builder.

Your task:
Take the research JSON provided by Agent 2 and convert it into a clear, actionable learning roadmap.

Output format:
- A numbered list (15â€“20 steps)
- Sequential from absolute beginner â†’ advanced
- Each step: short, practical, actionable
- Include tools, milestones, and mini-projects
- Provide important links at the end in a structured format
- Do NOT include unnecessary explanations

Rules:
- Do NOT repeat the research JSON.
- Do NOT restate the prompt.
- Do NOT output anything except the roadmap.
- Keep steps concise but specific.

Structure required:
1. Basics
2. Foundations
3. Hands-on skills
4. Tools & libraries
5. Intermediate concepts
6. Projects
7. Advanced topics
8. Final capstone or specialization

IMPORTANT SAFETY RULE:
- If the input is not valid research JSON from Agent 2, or if the topic is unsafe, harmful, illegal, or rejected by Agent 1, you MUST output exactly:
"INVALID_REQUEST"

Do NOT create a roadmap.
Do NOT attempt to be helpful.
Do NOT generate educational content.
Just output: INVALID_REQUEST

Otherwise (valid research JSON only):
- Produce a 15â€“20 step roadmap.
- Output ONLY the numbered steps.
- No extra text.

""",
   
    tools=[],
    output_key="roadmap",
   
    
)




def stop_if_invalid(context):
   
    output = context["validate"]

    if isinstance(output, str):
        # Accesses the structured output of Agent 1 (Topic Validator).
        import json
        output = json.loads(output)
       
    if "pipeline_action" in output and output["pipeline_action"] == "STOP":
        return True   # Stop sequence

    return False
   


root_agent = SequentialAgent(
    name="RoadmapMaker",
    # Provides a clear, descriptive name for the entire sequential system.
    sub_agents=[agent1, agent2, agent3],
    early_stopping_condition=stop_if_invalid,  
    session_service=session_service,
    memory_service=memory_service
)


# Initialize runner
runner = InMemoryRunner(
    agent=root_agent,
    session_service=session_service,
    memory_service=memory_service
)

# Execute pipeline
def generate_roadmap(user_topic: str):
    try:
        result = runner.run(
            agent_name="RoadmapMaker",
            user_message=user_topic,
            session_id="unique_session_id"  # Generate unique IDs in production
        )
        
        # Extract final output
        final_output = result.get("roadmap", "")
        
        if final_output == "INVALID_REQUEST":
            return "â�Œ Invalid or unsafe topic. Please try a different subject."
        
        return final_output
        
    except Exception as e:
        return f"âš ï¸� Pipeline error: {str(e)}"

# Example usage
topic = "Data Visualization using Python"
roadmap = generate_roadmap(topic)
print(roadmap)


runner = InMemoryRunner(agent=root_agent)
# Instantiates the final execution runner, linking it to the SequentialAgent.
response = await runner.run_debug(
    "data visualisation using python"
)

