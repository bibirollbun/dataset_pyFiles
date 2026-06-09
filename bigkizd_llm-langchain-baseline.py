!pip install --no-index --no-deps /kaggle/input/langchainhuggingface/bitsandbytes-0.45.2-py3-none-manylinux_2_24_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/langchainhuggingface/langchain_huggingface-0.1.2-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/langchainhuggingface/langchain_community-0.3.17-py3-none-any.whl





import os
import gc
import warnings

import numpy as np
import pandas as pd
import polars as pl

import torch
from transformers import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from operator import itemgetter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain.agents import Tool
from langchain.tools.retriever import create_retriever_tool

warnings.simplefilter('ignore')
set_seed(2025)
if torch.cuda.is_available():
    print("Cuda is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    


df_test = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
df_sub = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv")
df_sub.head()


model_name = '/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/1'
model_name = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b/1"
# llm = LLM(
#     model_name,
#     dtype="half",            
#     #max_num_seqs=128,            -> Changed this
#     max_model_len=4096,#4096*10,         
#     trust_remote_code=True,     
#     tensor_parallel_size=2,      
#     gpu_memory_utilization=0.96, 
# )
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True,
    quantization_config=quantization_config,
    # torch_dtype = torch.bfloat16,
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens = int(1024*1.5))


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
print_gpu_memory()


model = HuggingFacePipeline(pipeline=pipe)
model


from langchain.agents import AgentExecutor, create_react_agent
def get_simple_chain():
    prompt = PromptTemplate.from_template("""
You are an expert writer specializing in well-structured essays. 
Given the topic: "{question}", follow this step-by-step reasoning to generate an insightful essay:

1. **Understanding the Topic**: Briefly explain what the topic is about.
2. **Breaking Down Key Ideas**: Identify the main points to discuss.
3. **Logical Argumentation**: Develop a chain of thought reasoning to support each point.  Explain your reasoning step by step.
4. **Conclusion**: Summarize the key insights effectively.

Now, write the full essay following this structured approach.  Clearly label each section (Understanding the Topic, Breaking Down Key Ideas, Logical Argumentation, Conclusion).
    """)
    chain = (
        prompt
        | model
        | StrOutputParser()
    )
    return chain

    
simple_chain = get_simple_chain()
response = simple_chain.invoke({"question": "What is demand forecasting?"})
print(response)





@tool
def chain_of_thought(question: str) -> str:
    """Chain of Thought tool.  Used to perform reasoning and answer complex questions."""
    prompt = PromptTemplate.from_template("""
Question: {question}

Let's think step by step. Explain your reasoning process in detail, outlining all steps taken to arrive at the answer.  Then, provide the final answer, clearly marked.
""")
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({"question": question})
    return response
    
    

def get_cot_chain():
    prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}    
""")
    chain_of_thought_tool = Tool(
        name="ChainOfThought",
        func = chain_of_thought,
        description = "Chain of Thought tool.  Used to perform reasoning and answer complex questions."
    )
    tools = [chain_of_thought_tool]
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=True, handle_parsing_errors=True)

    chain = (
        {"input": itemgetter("question")}
        | agent_executor
        | itemgetter("output")
        | StrOutputParser()
    )
    return chain

chain = get_cot_chain()

response = chain.invoke({"question": "What is demand forecasting?"})

print(f"Response: {response}")


cot_chain = get_cot_chain()


def predict_for_question(topic):
    response = None
    try:
        response = cot_chain.invoke({"question": topic})
    except:
        response = simple_chain.invoke({"question": topic})
    finally:
        if response is None:
            response = "Nothing here"
    return response


df_test["essay"] = df_test["topic"].apply(lambda x: predict_for_question(x))
df_test.drop(columns = ["topic"]).to_csv("submission.csv", index = False)


pd.read_csv("submission.csv").head()




