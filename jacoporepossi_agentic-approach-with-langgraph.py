!pip install -Uq langgraph langchain-community langchain-huggingface


!pip install -q -i https://pypi.org/simple/ bitsandbytes


import pandas as pd
import numpy as np

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
from IPython.display import Image, display, Markdown
import re

from pprint import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from kaggle_secrets import UserSecretsClient


user_secrets = UserSecretsClient()
token = user_secrets.get_secret("hf_token")

model_id = "mistralai/Ministral-8B-Instruct-2410"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config, 
    token=token
)


base_prompt = """
Answer the following mathematical problem as pricise as you can. You have access to the following tool:
- calculator: useful for getting the result of a math expression. The input to this tool should be a valid mathematical python expression that could be executed by exec or eval.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and state it clearly. Consider also past Thought before planning the next one
Action: the tool you should use, in this case [calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (only a number)

Let's begin!

Question: """

problem = "20 cars of different colors are parked in a row. The black car is 16th from the right, and the white car is 11th from the left. How many cars are between the black and white cars?\n"


class State(TypedDict):
    prompt: str
    output: str
    tool_output: str


def generator(state: State):
    """
    Processes the current state by generating a model response using the tokenizer and model up to 'Observation'
    """
    # Handles the initial step, when we have no tool_output but only the problem to append to the base prompt
    if state['tool_output'] is None:
        state['prompt'] = state['prompt'] + state['output']
        messages = [
            {"role": "user", "content": state['prompt'] + state['output']}
        ]
        
    else:
    # Concatenate the previous output and the tool_ouput back into the model's response
        state['prompt'] = state['prompt'] + "\n" + state['output'] + ": " + state['tool_output']
        messages = [
            {"role": "user", "content": state['prompt']}
        ]
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    inputs = tokenizer(tokenized_chat, return_tensors="pt").to(device)
    # Generate a response until the string Observation is reached. This means that we should see an Action Input before it
    outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.3, top_k=20, top_p=0.2, stop_strings='Observation', tokenizer=tokenizer) 
    output_clean = tokenizer.decode(outputs[0])[len(tokenized_chat)+3:]
    
    return {"output": output_clean, "prompt": state['prompt']}

def router(state: State):
    """
    Determines the next node in the graph based on patterns in the state output (either calculator of final_answer)
    """
    tool_match = re.search(r'Action: (.*?)(?=Action Input)', state['output'], re.DOTALL)
    final_answer_match = re.search(r'Final Answer', state['output'], re.DOTALL)

    tool = tool_match.group(1).strip() if tool_match else "no_tool"
    final_answer = final_answer_match.group().strip() if final_answer_match else "no_final_answer"
    if 'calculator' in tool:
        return 'coder'
    elif final_answer != "no_final_answer":
        return 'end'
    else:
        return 'error'

def coder(state: State):
    """
    Evaluates a mathematical expression or other simple executable action extracted from the state's output.
    """
    match = re.search(r'Action Input: (.*?)(?=Observation)', state['output'], re.DOTALL)
    action_input_text = match.group(1).strip().strip('"') if match else "FAILED"
    try:
        coder_output = eval(action_input_text)
    except Exception as e:
        coder_output = "This is not something that could be executed by a simple calculator! You need to pass basic mathematical expressions that can be executed by a calculator"

    return {"tool_output": str(coder_output)}


graph_builder = StateGraph(State)

graph_builder.add_node('generator', generator)
graph_builder.add_node('coder', coder)

graph_builder.add_edge(START, "generator")
graph_builder.add_edge("coder", "generator")
graph_builder.add_conditional_edges('generator', router, {'coder': 'coder', 'error': END, 'end': END})

config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


display(Image(graph.get_graph().draw_mermaid_png()))


for chunk in graph.stream(
    {"output": problem, "prompt": base_prompt, 'tool_output': None},
    config=config,
    stream_mode="updates",
):
    pprint(chunk)
    print('#'*100)


output = graph.invoke({"output": problem, "content": base_prompt, 'tool_output': None}, config=config)


print(output['output'])


for state in graph.get_state_history(config):
    if not state.metadata['writes']:
        continue
    if state.metadata['writes'].get('generator', False):
        print(
"""
===============================================
AT STEP {}, GENERATOR'S TURN:

PROMPT
{}

OUTPUT
{}
""".format(state.metadata.get('step'), state.values.get('prompt'), state.values.get('output'))      
    )

    elif state.metadata['writes'].get('coder', False):
        print("===============================================\n\nCODER OUTPUT AT STEP {} = {}".format(state.metadata['step'], state.values['tool_output'])      
    )


model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
tokenizer_deep = AutoTokenizer.from_pretrained(model_id, token=token)

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model_deep = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config, 
    token=token
)


class State(TypedDict):
    prompt: str
    problem: str
    output_1: str
    output_2: str
    combined_output: str
   
    
def react_agent(state: State):
    """
    Processes the problem with a basic ReACT workflow
    """
    messages = [
            {"role": "user", "content": state['prompt'] + state['problem']}
        ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    inputs = tokenizer(tokenized_chat, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1200, do_sample=True, temperature=0.3, top_k=20, top_p=0.2, tokenizer=tokenizer) 
    output_clean = tokenizer.decode(outputs[0])[len(tokenized_chat)+3:]
    
    return {"output_1": output_clean}

def cot_agent(state: State):
    """
    Processes the problem with a basic CoT prompt
    """
    messages = [
            {"role": "user",
             "content": """
 Given the following mathematical problem, solve it the best you can. Always explain your thoughts step by step until reaching the final solution.
 
 PROBLEM
 {}
 """.format(state['problem'])}
        ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_chat = tokenizer_deep.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    inputs = tokenizer_deep(tokenized_chat, return_tensors="pt").to(device)
    outputs = model_deep.generate(**inputs, max_new_tokens=2000, do_sample=True, temperature=0.6, top_k=20, top_p=0.3, tokenizer=tokenizer) 
    output_clean = tokenizer_deep.decode(outputs[0])[len(tokenized_chat):]
    
    return {"output_2": output_clean}

def aggregator(state: State):
    """
    Aggregate the results and come up with the final solution
    """
    messages = [
            {"role": "user",
             "content": """
 Given the following [PROBLEM] and two [OUTPUT], return the result you consider the most accurate (double check the answer).
 You can also come up with a different solution if none of the two seem correct but always explain your reasonings step by step and the strategy you chose.

 Let's begin!
 
 ======== PROBLEM =========
 {}
 
 ====== OUTPUT #1 =========
 {}
 
 ====== OUTPUT #2 =========
 {}""".format(state['problem'], state['output_1'], state['output_2'])}
        ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    inputs = tokenizer(tokenized_chat, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.3, top_k=20, top_p=0.2, tokenizer=tokenizer) 
    output_clean = tokenizer.decode(outputs[0])[len(tokenized_chat)+3:]
    return {"combined_output": (tokenized_chat, output_clean)}


graph_builder = StateGraph(State)

graph_builder.add_node('react_agent', react_agent)
graph_builder.add_node('cot_agent', cot_agent)
graph_builder.add_node('aggregator', aggregator)

graph_builder.add_edge(START, "react_agent")
graph_builder.add_edge(START, "cot_agent")
graph_builder.add_edge("react_agent", "aggregator")
graph_builder.add_edge("cot_agent", "aggregator")
graph_builder.add_edge("aggregator", END)

config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


display(Image(graph.get_graph().draw_mermaid_png()))


output = graph.invoke({"problem": problem, "prompt": base_prompt}, config=config)


print('INITIAL PROMPT\n', '#'*100)
print(output['combined_output'][0])
print('#'*100, '\nAGGREGATOR ANSWER')
print(output['combined_output'][1])

