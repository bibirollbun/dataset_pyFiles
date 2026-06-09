import os

try:
    # Try getting secret from Kaggle
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Gemini API key setup complete.")
except Exception as e:
    # Fallback for local development if env var is already set
    if "GOOGLE_API_KEY" in os.environ:
        print("✅ Gemini API key found in environment variables.")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    else:
        print(f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your secrets. Details: {e}")


from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.genai import types

print("✅ ADK components imported successfully.")


safety_analyst = Agent(
    name="SafetyAnalyst",
    model="gemini-2.5-flash-lite",
    instruction="""You are an expert content moderator. Analyze the given text for toxicity, bullying, hate speech, or harassment.
    
    Output a JSON object with the following fields:
    - is_toxic (boolean): True if the content is harmful, False otherwise.
    - category (string): The type of toxicity (e.g., "Bullying", "Hate Speech", "Insult", "Safe").
    - severity (string): "Low", "Medium", or "High".
    - reasoning (string): A brief, objective explanation of why it was flagged.
    
    Do not output markdown formatting, just the JSON object.""",
    output_key="safety_report"
)

print("✅ SafetyAnalyst created.")


empathy_coach = Agent(
    name="EmpathyCoach",
    model="gemini-2.5-flash-lite",
    instruction="""You are a kind and patient empathy coach for children. 
    You will receive a {safety_report} about a message a child tried to send.
    
    Your goal is to explain to the child WHY their message might be hurtful, using age-appropriate language (aim for 10-12 year olds).
    Focus on feelings and impact. Do not lecture; guide them to understand the other person's perspective.
    Keep it short (2-3 sentences).""",
    output_key="empathy_explanation"
)

print("✅ EmpathyCoach created.")


constructive_mediator = Agent(
    name="ConstructiveMediator",
    model="gemini-2.5-flash-lite",
    instruction="""You are a helpful communication assistant. 
    You will receive a toxic message and a safety report.
    
    Your task is to suggest 3 alternative ways to express the user's underlying feeling or point, but in a constructive, polite, and safe way.
    If the user was just being mean without a point, suggest ways to disengage or say nothing.
    Format the output as a numbered list.""",
    output_key="constructive_alternatives"
)

print("✅ ConstructiveMediator created.")


guardian_manager = Agent(
    name="GuardianManager",
    model="gemini-2.5-flash-lite",
    instruction="""You are the SafeSpace Guardian Manager. Your goal is to moderate and mentor users.
    
    Follow this workflow strictly:
    1.  **Analyze**: ALWAYS start by calling the `SafetyAnalyst` tool to evaluate the user's message.
    2.  **Decide**:
        - IF the `SafetyAnalyst` reports `is_toxic: true`:
            a. Call the `EmpathyCoach` tool to get an explanation.
            b. Call the `ConstructiveMediator` tool to get alternatives.
            c. **Final Response**: Combine the outputs into a single helpful message for the user. Structure it as:
               "⚠️ Hold on! That message seems [Category]."
               "[Empathy Coach Explanation]"
               "Here are some better ways to say it:"
               "[Constructive Alternatives]"
        
        - IF the `SafetyAnalyst` reports `is_toxic: false`:
            a. Do NOT call the Coach or Mediator.
            b. **Final Response**: Simply reply "✅ Message looks safe! Sent."
    """,
    tools=[
        AgentTool(safety_analyst),
        AgentTool(empathy_coach),
        AgentTool(constructive_mediator)
    ]
)

print("✅ GuardianManager created.")


runner = InMemoryRunner(agent=guardian_manager)


response = await runner.run_debug("You are so stupid, no one likes you!")


response = await runner.run_debug("I really like your drawing, it's cool!")


response = await runner.run_debug("I hate this game, you guys are playing like trash.")

