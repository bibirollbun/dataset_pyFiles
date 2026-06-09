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


!pip install crewai langchain langchain-community langchain-openai openai python-dotenv


import os

os.environ["OPENAI_API_KEY"] = ""sk-XXXXXXXX""




from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# -------------------------------
# Agents
# -------------------------------

intent_agent = Agent(
    name="Intent Parser",
    role="Extracts travel preferences from user text",
    goal="Turn user text into JSON with destination, dates, budget, interests",
    backstory="Expert travel planner",
    llm=llm
)

flight_agent = Agent(
    name="Flights Recommender",
    role="Generate flight options",
    goal="Return 3 low-cost flight options",
    backstory="Experienced travel deals hunter",
    llm=llm
)

hotel_agent = Agent(
    name="Hotels Recommender",
    role="Recommend hotels",
    goal="Return 3 hotels with ratings and nightly cost",
    backstory="Budget hotel expert",
    llm=llm
)

itinerary_agent = Agent(
    name="Trip Planner",
    role="Make itineraries",
    goal="Generate a realistic 3-day itinerary",
    backstory="Travel influencer who knows hidden gems",
    llm=llm
)

# -------------------------------
# Tasks
# -------------------------------

task_intent = Task(
    agent=intent_agent,
    description="Parse user input into JSON",
    expected_output="Valid JSON only"
)

task_flights = Task(
    agent=flight_agent,
    description="Generate 3 flight options in JSON list",
    expected_output="JSON of flight options"
)

task_hotels = Task(
    agent=hotel_agent,
    description="Recommend 3 good hotels with rating + price",
    expected_output="JSON list of hotels"
)

task_itinerary = Task(
    agent=itinerary_agent,
    description="Build a 3-day itinerary using flights and hotels",
    expected_output="Markdown formatted itinerary"
)

# -------------------------------
# CREW
# -------------------------------
crew = Crew(
    agents=[intent_agent, flight_agent, hotel_agent, itinerary_agent],
    tasks=[task_intent, task_flights, task_hotels, task_itinerary],
    verbose=True
)



 from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)



intent_agent = Agent(
    name="IntentParser",
    role="Extract JSON travel data",
    goal="Parse user text into JSON with destination, budget, dates and interests.",
    backstory="Experienced travel agent",
    llm=llm
)

flight_agent = Agent(
    name="FlightExpert",
    role="Generate flight suggestions",
    goal="Suggest 3 cheap flights.",
    backstory="Airfare deal hunter",
    llm=llm
)

hotel_agent = Agent(
    name="HotelExpert",
    role="Recommends hotels",
    goal="Suggest 3 budget-friendly hotels.",
    backstory="Hotel pricing analyst",
    llm=llm
)

itinerary_agent = Agent(
    name="TripPlanner",
    role="Generate itinerary",
    goal="Make a realistic schedule of activities.",
    backstory="Travel influencer who knows hidden gems",
    llm=llm
)



task1 = Task(
    description="Extract JSON travel preferences",
    agent=intent_agent,
    expected_output="Only JSON (destination, dates, budget, interests)"
)

task2 = Task(
    description="Recommend 3 flight options in JSON",
    agent=flight_agent,
    expected_output="List of flights JSON"
)

task3 = Task(
    description="Recommend 3 hotel options with rating + price",
    agent=hotel_agent,
    expected_output="List of hotels JSON"
)

task4 = Task(
    description="Generate 3-day itinerary using flights and hotels",
    agent=itinerary_agent,
    expected_output="Markdown itinerary"
)



crew = Crew(
    agents=[intent_agent, flight_agent, hotel_agent, itinerary_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True
)



query = """
Plan a 4-day Bangkok trip for 2 people.
Budget 1200 USD.
Interests: street food, temples, nightlife.
Origin: New Delhi.
"""

result = crew.kickoff(inputs={"user_input": query})
print(result)





