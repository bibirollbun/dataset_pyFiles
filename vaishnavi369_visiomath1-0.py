import os 
from kaggle_secrets import UserSecretsClient 
try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY 
    print("âœ… GOOGLE API key setup complete.") 
except Exception as e: 
    print( f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}" )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent 
from google.adk.models.google_llm import Gemini 
from google.adk.runners import Runner 
from google.genai import types
from google.adk.runners import InMemoryRunner 
from google.adk.tools import google_search

print("âœ… ADK components imported successfully.")


retry_config = types.HttpRetryOptions( 
    attempts=5, # Maximum retry attempts 
    exp_base=7, # Delay multiplier 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)
print (" â–  successful â– ")


image_agent = Agent( 
    name="ImageAgent", 
    model=Gemini( model="gemini-2.5-flash-lite", 
                 retry_options=retry_config ),
    instruction="""You are specialized image search agent. You should search images for the theorem which user ask for, in graph form and, pick the easiest and flexibly understandable graphical picture for the theorem that should be easily understandable by the user. Once you got the image, fetch the url with the response.""", 
    tools=[google_search], output_key="images", ) 

print("âœ… image_agent created.")


finder_agent = Agent( 
    name="FinderAgent", 
    model=Gemini( model="gemini-2.5-flash-lite", 
                 retry_options=retry_config ), 
    instruction="""You are a specialized agent and when user asks about a theorem, you aim is to provide the desired theorem in a simpler way that user should understand easily and clearly. Use google_search tool for searching and returning a quality and clear answer to the user. Keep it concise unless the user asks for breif explanation. Use Latex for complex mathematical terms in the theorem. """, 
    tools=[google_search], output_key="findings", 
) 

print("âœ… finder_agent created.")


collector_agent = Agent( 
    name="CollectiveAgent", 
    model=Gemini( model="gemini-2.5-flash-lite",
                 retry_options=retry_config ), 
    instruction="""Display {findings} and {images} """, 
    tools=[google_search] 
) 

print("âœ… collector_agent created.")


final_agent= ParallelAgent( 
    name="FAgent", 
    sub_agents=[finder_agent, image_agent], 
) 
root_agent = SequentialAgent( 
    name="RootAgent", 
    sub_agents=[final_agent, collector_agent], 
) 

print("âœ… Parallel and Sequential Agents created.")


runner = InMemoryRunner(agent =root_agent ) 
response = await runner.run_debug("what is pythogoras theorem with picture to understand")

