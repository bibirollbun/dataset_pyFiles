try:
    from kaggle_secrets import UserSecretsClient
    import os
    
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("Kaggle Secrets setup successfully.")

except Exception as e:
    print(
        f"Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' & 'YOUTUBE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import google_search, AgentTool, FunctionTool
import googleapiclient.discovery
from google.genai import types

print("ADK libraries loaded successfully.")


USER_ID = "default"
APP_NAME = "default"
SESSION = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

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


print("✅ Helper functions defined.")


workout_planner = Agent(
    name="WorkoutPlanner",
    model="gemini-2.5-flash-lite",
    description="This agent plans exercises.",
    instruction="You are workout coach. Assess user biometrics and create an exercise plan. Ensure that the plan is not too intense and suggested exercises are safe. Use the google search tool to find the best workout plan and to include reference YouTube short videos.",
    tools=[google_search],
)


diet_planner = Agent(
    name="DietPlanner",
    model="gemini-2.5-flash-lite",
    description="This agent creates a meal plan.",
    instruction="You are an expert nutritionist. Assess user biometrics and goals to create a meal plan respecting the dietary preferences. Use google search tool to find recipes and to include reference YouTube short videos.. Include dietary information for every meal like calories, amount of protein, carbohydrates and fats.",
    tools=[google_search],
)


root_agent = Agent(
    name="FitnessCoordinator",
    model="gemini-2.5-flash-lite",
    instruction='''
    Use the workout_planner and diet_planner tools to generate a fitness timetable for the user based on their profile.
    ''',
    tools=[AgentTool(workout_planner), AgentTool(diet_planner)]
)


user_profile = """
- Gender: Male
- Age: 25 years
- Height: 5ft 8in
- Weight: 133lbs
- Workout schedule: Monday to Saturday, 5pm - 6:30pm
- Dietary preferences: Non-vegetarian (white-meats only)
- Medical conditions: None
- Goal: Gain muscle, lose fat.
"""


db_url = "sqlite:///fitness_coach.db"
session_service = DatabaseSessionService(db_url=db_url)

runner = Runner(agent = root_agent, app_name=APP_NAME, session_service=session_service)


await run_session(runner, 
                  [
                    f"Generate workout and diet routine for the user profile:\n{user_profile}",
                    "What is the exercise plan for Monday?"
                  ],
                  "test-db-session-01"
                 )


await run_session(runner, 
                  [
                    "What is are my biometrics?"
                  ],
                  "test-db-session-01"
                 )

