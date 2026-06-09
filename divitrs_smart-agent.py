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
from crewai import Agent, Task, Crew, Process
from crewai_tools import GoogleSearchTool
from dotenv import load_dotenv

# --- PREREQUISITES AND SETUP ---

# 1. Install required libraries:
#    pip install crewai 'crewai[google-search]' google-genai python-dotenv

# 2. Create a file named .env in the same directory and add your API keys:
#    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
#    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
#    GOOGLE_CSE_ID="YOUR_GOOGLE_CUSTOM_SEARCH_ENGINE_ID" 
# (You must enable the Google Search API and create a Custom Search Engine.)

# Load environment variables from the .env file
load_dotenv()

# Check for API keys before proceeding
if not all([os.getenv('GEMINI_API_KEY'), os.getenv('GOOGLE_API_KEY'), os.getenv('GOOGLE_CSE_ID')]):
    print("â�Œ Error: Please ensure GEMINI_API_KEY, GOOGLE_API_KEY, and GOOGLE_CSE_ID are set in your .env file.")
    exit()

# --- 1. Define the Tool ---
# The GoogleSearchTool enables agents to find real-time, external data.
search_tool = GoogleSearchTool()

# --- 2. Define the Agents (The Multi-Agent System) ---

## Agent 1: The Researcher (Uses the Tool)
research_agent = Agent(
    role='Senior Research Analyst',
    goal='Identify and gather the most crucial, recent data and facts about the user\'s topic from the web.',
    backstory='An expert in fast, accurate, and relevant data retrieval. You specialize in using search engines to find the needle in the digital haystack.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm='gemini-2.5-flash'
)

## Agent 2: The Analyzer (Synthesizes Data)
analyzer_agent = Agent(
    role='Data & Impact Analyst',
    goal='Analyze the raw search results and synthesize them into a structured report outlining key findings, trends, and potential impacts.',
    backstory='A meticulous analyst who turns raw data into actionable insights, focusing on structure, clarity, and depth of understanding.',
    verbose=True,
    allow_delegation=False,
    llm='gemini-2.5-flash'
)

## Agent 3: The Reporter (Final Output Generator)
reporter_agent = Agent(
    role='Executive Report Writer',
    goal='Take the analyzed insights and craft a final, polished, executive-summary-style report suitable for a professional audience.',
    backstory='A skilled communicator who excels at presenting complex information clearly, concisely, and persuasively.',
    verbose=True,
    allow_delegation=False,
    llm='gemini-2.5-flash'
)

# --- 3. Define the Tasks (The Sequential Workflow) ---

## Task 1: Research and Data Gathering
task_research = Task(
    description=(
        "Search the web for the latest, most impactful information regarding the topic: '{topic}'."
        "Focus on key statistics, recent developments, and expert opinions."
        "The output should be a collection of relevant, raw search snippets."
    ),
    expected_output='A bulleted list of raw, factual search snippets.',
    agent=research_agent
)

## Task 2: Analysis and Synthesis
task_analyze = Task(
    description=(
        "Review the raw snippets provided by the Researcher."
        "Synthesize this information into three main sections: **Key Findings**, **Emerging Trends**, and **Potential Impacts**."
        "Focus on connecting the data points into coherent insights."
    ),
    expected_output='A well-structured document with the three required sections and concise, insightful paragraphs.',
    agent=analyzer_agent,
    context=[task_research] # Passes the output of T1 to T2
)

## Task 3: Final Report Generation
task_report = Task(
    description=(
        "Use the analyzed and synthesized insights to write the final, executive-level report."
        "The report MUST include a catchy **Title**, a brief **Executive Summary**, and the content from the three sections provided by the Analyzer."
        "Ensure professional tone and polish."
    ),
    expected_output='A complete, single-page, professional report with a Title, Summary, and structured content.',
    agent=reporter_agent,
    context=[task_analyze] # Passes the output of T2 to T3
)


# --- 4. Create the Crew and Run the Process ---

if __name__ == "__main__":
    
    # Get the topic from the user
    topic_of_interest = input("What topic would you like the AI Research Crew to investigate for your Capstone? ")

    # Instantiate the Crew
    research_crew = Crew(
        agents=[research_agent, analyzer_agent, reporter_agent],
        tasks=[task_research, task_analyze, task_report],
        process=Process.sequential, # The tasks run in order: 1 -> 2 -> 3 (Orchestration)
        verbose=2 # Shows the full execution details for debugging/demonstration
    )

    # Execute the workflow
    print("\n\nğŸš€ Launching the Research Crew...")
    crew_result = research_crew.kickoff(inputs={'topic': topic_of_interest})

    # --- 5. Print the Final Output ---
    print("\n\n################################################################")
    print("##               AI-GENERATED EXECUTIVE REPORT              ##")
    print("################################################################\n")
    print(crew_result)
    print("\n################################################################")
    print("##             CAPSTONE PROJECT COMPLETED                   ##")
    print("################################################################")

