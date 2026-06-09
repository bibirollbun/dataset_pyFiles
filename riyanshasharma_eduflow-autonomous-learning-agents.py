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


# IMPORTS 
import asyncio
import json
import os
import time
from typing import Dict, Any, List

import gradio as gr

# SIMPLE MCP BUS
class MCPBus:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn):
        self.tools[name] = fn

    async def call(self, name, payload):
        fn = self.tools[name]
        res = fn(payload)
        if asyncio.iscoroutine(res):
            res = await res
        return res

# TOOLS
def tool_save(payload):
    path = payload["path"]
    content = payload["content"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)
    return {"ok": True}
    
# tiny retriever dataset
DOCS = [
{"title": "Photosynthesis", "text": "Plants convert sunlight into energy using chlorophyll."},
{"title": "Cell", "text": "Cells are the basic unit of life containing organelles."},
{"title": "Mitosis", "text": "Mitosis is cell division producing two identical daughter cells through phases like prophase and metaphase."},
{"title": "Ecosystem", "text": "An ecosystem includes all living organisms interacting with their physical environment."},
{"title": "Gravity", "text": "Gravity is the force that attracts two bodies toward each other, keeping planets in orbit."},
{"title": "Nutrition", "text": "Nutrition involves the intake of food necessary for growth, repair, and energy in living organisms."},
{"title": "Water Cycle", "text": "The water cycle includes evaporation, condensation, precipitation, and collection."},
{"title": "Atoms", "text": "Atoms are the smallest units of matter made up of protons, neutrons, and electrons."},
{"title": "Energy", "text": "Energy comes in many forms including kinetic, potential, chemical, and thermal."},
{"title": "Digestive System", "text": "The digestive system breaks down food into nutrients the body can absorb and use."}
]

def tool_retrieve(payload):
    q = payload["query"].lower()
    hits = [d for d in DOCS if any(w in d["text"].lower() for w in q.split())]
    return {"results": hits[:3]}

# Register tools
mcp = MCPBus()
mcp.register("save", tool_save)
mcp.register("retrieve", tool_retrieve)

# MOCK LLM
class LLM:
    async def generate(self, prompt):
        if "quiz" in prompt.lower():
            return "Q1: What is photosynthesis?\nA: Process where plants make food using sunlight."
        return "(LLM Response) " + prompt[:120]

llm = LLM()

# AGENTS
class QuizAgent:
    async def run(self, topic):
        text = await llm.generate(f"create quiz on {topic}")
        await mcp.call("save", {"path": f"quiz_{topic}.json", "content": {"quiz": text}})
        return text

class ResearchAgent:
    async def run(self, query):
        r = await mcp.call("retrieve", {"query": query})
        response = await llm.generate(str(r))
        return response

class TutorAgent:
    paused = False

    async def run(self, items):
        logs = []
        for item in items:
            while self.paused:
                await asyncio.sleep(0.3)
            ans = await llm.generate(f"teach {item}")
            logs.append({"item": item, "teach": ans})
        return logs

quiz_agent = QuizAgent()
research_agent = ResearchAgent()
tutor_agent = TutorAgent()

# FRONTEND (GRADIO) 
async def run_quiz(topic):
    r = await quiz_agent.run(topic)
    return r

async def run_research(q):
    r = await research_agent.run(q)
    return r

async def run_tutor(topic_list):
    items = [t.strip() for t in topic_list.split(',')]
    r = await tutor_agent.run(items)
    return json.dumps(r, indent=2)

def toggle_pause():
    tutor_agent.paused = not tutor_agent.paused
    return "Paused" if tutor_agent.paused else "Running"

# Gradio App
with gr.Blocks() as app:
    gr.Markdown("# ðŸ“˜EduFlow: Autonomous Learning Agents")

    with gr.Tab("Quiz Agent"):
        t = gr.Textbox(label="Topic")
        out = gr.Textbox(label="Quiz Output")
        btn = gr.Button("Generate Quiz")
        btn.click(run_quiz, t, out)

    with gr.Tab("Research Agent"):
        q = gr.Textbox(label="Search Query")
        out2 = gr.Textbox(label="Research Summary")
        btn2 = gr.Button("Run Research")
        btn2.click(run_research, q, out2)

    with gr.Tab("Tutor Agent (Loop)"):
        topics = gr.Textbox(label="Comma separated topics")
        out3 = gr.Textbox(label="Teaching Output")
        btn3 = gr.Button("Start Tutor")
        pause_btn = gr.Button("Pause / Resume")
        btn3.click(run_tutor, topics, out3)
        pause_btn.click(toggle_pause, outputs=out3)

app.launch()

