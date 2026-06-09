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


!pip install google-generativeai serpapi



import google.generativeai as genai
import requests
from datetime import datetime
from kaggle_secrets import UserSecretsClient

# Load API key from Kaggle Secrets
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Use a valid model from your list
model = genai.GenerativeModel("models/gemini-2.5-flash")

memory = []



response = model.generate_content("Say hello in one short sentence.")
print(response.text)



# Load SerpAPI key
SERPAPI_KEY = user_secrets.get_secret("SERPAPI_KEY")

def web_search(query):
    url = "https://serpapi.com/search"
    params = {"q": query, "engine": "google", "api_key": SERPAPI_KEY}
    data = requests.get(url, params=params).json()

    results = data.get("organic_results", [])
    extracted = [
        {
            "title": r.get("title", ""),
            "link": r.get("link", ""),
            "snippet": r.get("snippet", "")
        }
        for r in results if r.get("snippet")
    ]
    return extracted[:5]   # Top 5 results



def research_agent(topic):
    print("\n [Research Agent] Collecting information...")
    search_results = web_search(topic)

    prompt = f"""
    You are an AI Research Specialist. Summarize the information below into a
    concise research report.

    Topic: {topic}

    Search Findings:
    {search_results}

    Include sections:
    - Key Insights
    - Benefits & Risks (if any)
    - Real-world Use Cases
    """

    response = model.generate_content(prompt)
    return response.text, search_results



def citation_agent(search_results):
    print("\n [Citation Agent] Generating citations...")

    sources = "\n".join([f"{item['title']} - {item['link']}" for item in search_results])

    prompt = f"""
    Convert the following into MLA style citations:

    {sources}
    """

    response = model.generate_content(prompt)
    return response.text



def multi_agent_research(topic):
    memory.append({"query": topic, "timestamp": str(datetime.now())})

    report, sources = research_agent(topic)
    citations = citation_agent(sources)

    final = f"""
     Research Report on: {topic}

    {report}

    Citations:
    {citations}

     Query Time: {memory[-1]["timestamp"]}
    """
    return final



topic = "Impact of Artificial Intelligence in Healthcare"
output = multi_agent_research(topic)
print(output)


