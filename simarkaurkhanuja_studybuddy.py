# Install & Imports (uncomment installs if needed)
# !pip install sentence-transformers faiss-cpu

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

MEMORY_FILE = 'agent_memory.json'
KNOWLEDGE_FILE = 'knowledge_base.csv'

# Create a tiny knowledge base if not exists
if not os.path.exists(KNOWLEDGE_FILE):
    df = pd.DataFrame([
        {"topic":"Kaggle","fact":"Kaggle is a platform for data science competitions, notebooks, and datasets."},
        {"topic":"Calculus","fact":"Integration by substitution is a method for evaluating integrals by changing variables."},
        {"topic":"Python","fact":"Python is a high-level programming language widely used in ML and data science."},
    ])
    df.to_csv(KNOWLEDGE_FILE, index=False)

print('Setup complete. Knowledge file:', KNOWLEDGE_FILE)


from dataclasses import dataclass, asdict

@dataclass
class MemoryItem:
    timestamp: str
    role: str
    content: str
    tags: List[str]

@dataclass
class ToolResult:
    tool_name: str
    result: Any
    meta: Dict[str,Any] = None

@dataclass
class DecisionTrace:
    timestamp: str
    user_input: str
    chosen_tools: List[str]
    prompts: Dict[str,str]
    tool_results: List[ToolResult]
    final_answer: str

DECISION_LOG: List[DecisionTrace] = []


# Memory module
MAX_MEMORY_ITEMS = 50

def load_memory() -> List[MemoryItem]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            raw = json.load(f)
            return [MemoryItem(**item) for item in raw]
    return []

def save_memory(items: List[MemoryItem]):
    raw = [asdict(i) for i in items[-MAX_MEMORY_ITEMS:]]
    with open(MEMORY_FILE, 'w') as f:
        json.dump(raw, f, indent=2)

def append_memory(role: str, content: str, tags: Optional[List[str]] = None):
    if tags is None: tags = []
    mem = load_memory()
    mem.append(MemoryItem(timestamp=datetime.utcnow().isoformat()+'Z', role=role, content=content, tags=tags))
    save_memory(mem)

print('Memory module ready')



# Tools: calculator, summarizer, knowledge lookup
import math

def calculator_tool(expression: str) -> ToolResult:
    try:
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith('__')}
        allowed_names.update({'abs': abs, 'round': round})
        result = eval(expression, {'__builtins__':None}, allowed_names)
        return ToolResult(tool_name='calculator', result=str(result), meta={'input': expression})
    except Exception as e:
        return ToolResult(tool_name='calculator', result=f'Error: {e}', meta={'input': expression})

def summarizer_tool(text: str, max_words=80) -> ToolResult:
    words = text.split()
    if len(words) <= max_words:
        summary = text
    else:
        summary = ' '.join(words[:max_words]) + '...'
    return ToolResult(tool_name='summarizer', result=summary, meta={'orig_len': len(words)})

def knowledge_lookup(query: str, top_k=3) -> ToolResult:
    df = pd.read_csv(KNOWLEDGE_FILE)
    matches = df[df.apply(lambda r: query.lower() in (r['topic'] + ' ' + r['fact']).lower(), axis=1)]
    if matches.empty:
        facts = df['fact'].tolist()
    else:
        facts = matches['fact'].tolist()
    return ToolResult(tool_name='knowledge_lookup', result=facts[:top_k], meta={'query': query})

print('Tools ready')



# LLM wrapper (simulated by default)
def llm_generate(prompt: str, max_tokens=256) -> str:
    """Replace with real LLM API calls (Gemini/OpenAI). By default this returns a simulated reply."""
    if os.environ.get('USE_REAL_LLM') == '1':
        return '(LLM placeholder)'
    simulated = f"(SIMULATED LLM) Based on prompt: {prompt[:200]}"
    return simulated

print('LLM wrapper ready (simulated)')


# Intent classifier
def classify_intent(user_input: str) -> str:
    ui = user_input.lower()
    if any(w in ui for w in ['calculate', 'solve', '+', '-', '*', '/', 'evaluate']):
        return 'calculator'
    if any(w in ui for w in ['summarize', 'summary', 'tl;dr']):
        return 'summarizer'
    if any(w in ui for w in ['who', 'what', 'when', 'tell me about', 'explain', 'define']):
        return 'knowledge'
    if any(w in ui for w in ['plan', 'schedule', 'study', 'learn']):
        return 'planner'
    return 'llm'

print('Intent classifier ready')


# Agents (Planner, Retriever, QA)
def planner_agent(goal: str, constraints: Dict[str,Any]) -> ToolResult:
    weeks = constraints.get('weeks', 2)
    hours_per_week = constraints.get('hours_per_week', 6)
    topics = [t.strip() for t in goal.split(',') if t.strip()]
    if not topics:
        topics = [goal]
    total_days = weeks * 7
    schedule = []
    for i in range(min(len(topics), total_days)):
        schedule.append({'day': i+1, 'topic': topics[i % len(topics)], 'hours': round(hours_per_week/7,2)})
    return ToolResult(tool_name='planner', result={'weeks': weeks, 'schedule': schedule}, meta={'goal': goal, 'constraints': constraints})

def retriever_agent(query: str, prefer_videos: bool=False) -> ToolResult:
    kr = knowledge_lookup(query)
    results = kr.result
    if prefer_videos:
        results = [r + ' (video suggested)' for r in results]
    return ToolResult(tool_name='retriever', result=results, meta={'query': query, 'prefer_videos': prefer_videos})

def qa_agent(question: str, retrieved_facts: List[str]) -> ToolResult:
    prompt = f"Use these facts {retrieved_facts} to answer: {question}"
    answer = llm_generate(prompt)
    confidence = 0.85 if 'SIMULATED' not in answer else 0.6
    return ToolResult(tool_name='qa', result={'answer': answer, 'confidence': confidence}, meta={'question': question})

print('Agents ready')


# Orchestrator
def orchestrate(user_input: str, use_memory: bool=True) -> DecisionTrace:
    ts = datetime.utcnow().isoformat() + 'Z'
    intent = classify_intent(user_input)
    chosen_tools = []
    prompts = {}
    tool_results = []

    mem = load_memory()
    mem_context = ' '.join([m.content for m in mem[-10:]]) if mem and use_memory else ''

    if intent == 'calculator':
        chosen_tools = ['calculator']
        tr = calculator_tool(user_input)
        tool_results.append(tr)
        final_answer = tr.result

    elif intent == 'summarizer':
        chosen_tools = ['summarizer']
        to_summarize = user_input if len(user_input.split())>10 else mem_context
        tr = summarizer_tool(to_summarize)
        tool_results.append(tr)
        final_answer = tr.result

    elif intent == 'knowledge':
        chosen_tools = ['retriever','qa']
        kr = retriever_agent(user_input, prefer_videos=('video' in mem_context.lower()))
        tool_results.append(kr)
        qa = qa_agent(user_input, kr.result)
        tool_results.append(qa)
        final_answer = qa.result['answer']

    elif intent == 'planner':
        chosen_tools = ['planner']
        constraints = {'weeks':2, 'hours_per_week':6}
        tr = planner_agent(user_input, constraints)
        tool_results.append(tr)
        final_answer = json.dumps(tr.result, indent=2)

    else:
        chosen_tools = ['llm']
        prompt = f"Context: {mem_context}\nUser: {user_input}\nAnswer concisely."
        prompts['llm'] = prompt
        llm_out = llm_generate(prompt)
        tool_results.append(ToolResult(tool_name='llm', result=llm_out, meta={}))
        final_answer = llm_out

    append_memory('user', user_input, tags=[intent])
    append_memory('agent', str(final_answer)[:1000], tags=chosen_tools)

    trace = DecisionTrace(timestamp=ts, user_input=user_input, chosen_tools=chosen_tools, prompts=prompts, tool_results=tool_results, final_answer=final_answer)
    DECISION_LOG.append(trace)
    return trace

print('Orchestrator ready')


# Demo 1: Store profile in memory
append_memory('user', 'Profile: 2nd-year CS student, prefers video resources, 6 hours/week available', tags=['profile'])
print('Profile stored. Memory snapshot:')
for m in load_memory()[-5:]:
    print(m.timestamp, m.role, m.tags, '-', m.content)



# Demo 2: Planner
q = 'Integration, Substitution, Definite Integrals'
trace = orchestrate(q)
print('Tools used:', trace.chosen_tools)
print('Planner output:')
print(trace.final_answer)



# Demo 3: Retriever + QA
q = 'Explain substitution method in integrals'
trace = orchestrate(q)
print('Tools used:', trace.chosen_tools)
for tr in trace.tool_results:
    print('---', tr.tool_name)
    print(tr.result)



# Demo 4: Calculator
q = 'calculate 23*(15+2)'
trace = orchestrate(q)
print('Tools used:', trace.chosen_tools)
print('Answer:', trace.final_answer)


