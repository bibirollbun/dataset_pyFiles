# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


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


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)


def flight_booking(source:str,destination:str) -> str:
    
   
    #f"Unsupported currency pair: {base_currency}/{target_currency}"
    
    if source is not None and destination is not None:
        return f"flight booked from {source} to {destination} successfully", 
    else:
        return  "error in booking flight",
    
print("flight booked")
#print("flight details ",flight_booking("india","america"))


    



from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
flight_agent=LlmAgent(
    name="flightagent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    book flight by calling 'flight_booking' 
    print only 'flight_booking' tool message """,
    tools=[flight_booking],
     
)
print("flight booking agent created using flight_booking tool")

    



def hotel_booking(city:str, name:str) -> str:
    if city is not None and name is not None:
        return f"hotel {name} is booked in city {city} successfully",
    else:
        return f"error occured in hotel booking"
print("hotel booked")        


hotel_agent=LlmAgent(
    name="hotelagent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
                book the hotel using 'hotel_booking' tool.
                print only 'hotel_booking' tool message 
                """,
    tools=[hotel_booking],           
)
print("hotel_agent called")


def tourisim_booking(beach:str, monument:str,museum:str)->str:
    if beach is not None and monument is not None and museum is not None:
        return f"tourist places booked are {beach} , {monument},{museum} successfully",
    else:
        return "error in booking tourist place"
print("tourist place booked")
#print("tourism", tourisim_booking("marina","church","Alex"))


tourist_agent=LlmAgent(
    name="touristagent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=""" 
            book tourist places using 'tourism_booking' tool
            print only 'tourism_booking' tool message """,
    tools=[tourisim_booking]
)
print("tourisim agent created")


booking_agent=SequentialAgent(
    name="seqagent",
    sub_agents=[flight_agent,hotel_agent, tourist_agent]
)
print("sequential")


from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import nest_asyncio
import asyncio
import datetime

session_service=InMemorySessionService()
session1=await session_service.create_session(
    app_name="booking_app",
    user_id="userid",
    session_id="sessionid"
    
   
    )
print("session started")

    
runner=Runner(
    app_name="booking_app",
    agent=booking_agent,
    session_service=session_service
    )


async def run_agent(query):
    
    content = types.Content(role="user", parts=[types.Part(text=query)])
    #await session_service.save_event(app_name="booking_app", user_id="userid", session_id=session.id, content=content)

    result=""
    async for event in runner.run_async(
        user_id="userid",
        session_id=session1.id,
        new_message=content
        
        ):
         
         if event.content and event.content.parts:
            for part in event.content.parts:
                if  part.text:
                    
                    print(" Agent  :", part.text)
                    print(end="\n")
                if getattr(part, "tool_call", None):
                    print("Tool call:", part.tool_call.model_name)

                if getattr(part, "tool_response", None):
                    print("Tool response:", part.tool_response.output)
                       

            
        
    return ' '.join(result)
            #print(type(event.content.parts[0].text))
            #print(len(event.content.parts[0].text))
            #show_python_code_and_result(event.content.parts[0].text)
async def booking_a2a():
        
    
    text="book a flight from america to india, book taj hotel in india and book tourist places marina beach,redfort,salarjung"  
    print(" User :", text)
    #booking=asyncio.run(main())
    booking=await run_agent(text)
    #print(" booking details1  :", booking, end="\n")
    print(end="\n")
    text2=text1="book flight from uk to india , book krishna hotel in india , book tourist places like marina beach , tajmahal , salarjung museum"
    print("User  :",text2)
    booking2=await run_agent(text2)
    #print(" booking details2  :", booking2)
    print(end="\n")

    new_session=await session_service.get_session(
        app_name="booking_app",
        user_id="userid",
        session_id="sessionid"
        
        
    )
    if new_session and new_session.events:


        for e in new_session.events:
    
    
            timestamp=datetime.datetime.now()
            
            for part in e.content.parts or []:
                    if part.text:
                        print(f"[{timestamp}]  : {part.text}")
                    if  hasattr(part,"tool_call") and part.tool_call is not None:
                        print(f"[{timestamp}] : TOOL CALL : {part.tool_call.model_name}")
                    if hasattr(part, "tool_response") and part.tool_response is not None:
                        print(f"[{timestamp}]  TOOL RESPONSE : {part.tool_response.output}")
            
    else:
        print("no session found")



await booking_a2a()    
    





