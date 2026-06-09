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


openai>=1.0.0
python-dotenv
requests


import requests

def web_search(query: str) -> str:
    """
    Dummy internet search tool (replace with real API).
    """
    return f"[Search result for '{query}']: This is example data. Replace with real API."


def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception:
        return "Error: invalid expression"


import json
from openai import OpenAI
from tools.web_search import web_search
from tools.calculator import calculator

client = OpenAI()

# TOOL REGISTRY
TOOLS = {
    "web_search": web_search,
    "calculator": calculator
}


# MAIN AGENT LOOP
def run_agent(message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or gpt-5.1 when available
        messages=[
            {"role": "system", "content": "You are an AI Agent with tool use ability."},
            {"role": "user", "content": message}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the internet",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]



mport tkinter as tk
from tkinter import scrolledtext
from agent import run_agent

def send():
    user_input = entry.get()
    chat_window.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    response = run_agent(user_input)
    chat_window.insert(tk.END, f"Agent: {response}\n\n")

root = tk.Tk()
root.title("AI Agent GUI")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
chat_window.pack()

entry = tk.Entry(root, width=70)
entry.pack()

send_button = tk.Button(root, text="Send", command=send)
send_button.pack()

root.mainloop()


rom fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent

app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/ask")
def ask_agent(payload: Query):
    result = run_agent(payload.message)
    return {"response": result}


rag/
│── ingest.py
│── retrieve.py
│── vector_store.faiss


import faiss
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "AI agents can use tools.",
    "FAISS is a vector database.",
    "Retrieval-Augmented Generation improves factuality."
]

vectors = model.encode(documents)
index = faiss.IndexFlatL2(384)
index.add(vectors)

faiss.write_index(index, "vector_store.faiss")

with open("docs.json", "w") as f:
    json.dump(documents, f)

print("Ingest complete.")


import faiss, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vector_store.faiss")
docs = json.load(open("docs.json"))

def retrieve(query):
    qv = model.encode([query])
    D, I = index.search(qv, 3)
    return [docs[i] for i in I[0]]


from agent import run_agent

def specialist_math(query):
    return f"[Math Specialist] {run_agent(query)}"

def specialist_search(query):
    return f"[Search Specialist] {run_agent(query)}"

def orchestrator(query):
    if "calculate" in query or any(x in query for x in ["+", "-", "*", "/"]):
        return specialist_math(query)
    return specialist_search(query)

if _name_ == "_main_":
    q = input("Ask multi-agent system: ")
    print(orchestrator(q))


from openai import OpenAI
client = OpenAI()

agent = client.agents.create(
    name="MyToolAgent",
    model="gpt-4o-mini",
    instructions="You are a tool-using agent.",
    tools=[
        {"type": "function", "function": {
            "name": "calculator",
            "description": "Perform math",
            "parameters": {"type": "object","properties":{"expression":{"type":"string"}}}
        }},
        {"type": "function", "function": {
            "name": "web_search",
            "description": "Search web",
            "parameters": {"type": "object","properties":{"query":{"type":"string"}}}
        }}
    ]
)

session = client.sessions.create(agent_id=agent.id)

msg = client.sessions.messages.create(
    session_id=session.id,
    role="user",
    content="Search AI agents and calculate 3*3"
)

print(msg.output_text)


pip install langchain-community langchain-openai


from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from tools.web_search import web_search
from tools.calculator import calculator

llm = ChatOpenAI(model="gpt-4o-mini")

tools = [
    Tool(
        name="web_search",
        func=web_search,
        description="Search the internet"
    ),
    Tool(
        name="calculator",
        func=calculator,
        description="Basic math calculations"
    )
]

agent = initialize_agent(
    tools, 
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

if _name_ == "_main_":
    print(agent.run("Search AI news and calculate 12*7"))

