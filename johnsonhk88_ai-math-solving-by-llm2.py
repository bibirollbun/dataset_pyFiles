# !pip install --target=/kaggle/working vllm bitsandbytes -U
# !rm -rf /kaggle/working/ray*





installDir = "/kaggle/input/universal-llm-install-package2/V7" #"/kaggle/input/universal-llm-install-package2/V7"
# installDir2 ="/kaggle/input/deepeval-open-source-llm-evaluation-framework"
!pip install transformers --no-index --no-deps --find-links=file://{installDir}/transformers-4.45.2-py3-none-any.whl
!pip install -U accelerate --no-index --no-deps --find-links=file://{installDir}/accelerate-1.0.1-py3-none-any.whl
!pip install -U trl --no-index --no-deps --find-links=file://{installDir}/trl-0.11.4-py3-none-any.whl
!pip install -U peft --no-index --no-deps --find-links=file://{installDir}/peft-0.14.0-py3-none-any.whl
!pip install  bitsandbytes --no-index --no-deps --find-links=file://{installDir}/bitsandbytes-0.45.0-py3-none-manylinux_2_24_x86_64.whl

!pip install -U langchain --no-index  --no-deps --find-links=file://{installDir}/langchain-0.3.3-py3-none-any.whl
!pip install -U langchain_core --no-index  --no-deps --find-links=file://{installDir}/langchain_core-0.3.12-py3-none-any.whl
!pip install -U langchain_text_splitters  --no-index  --no-deps  --find-links=file://{installDir}/langchain_text_splitters-0.3.0-py3-none-any.whl
!pip install -U langchain_community  --no-index  --no-deps   --find-link=file://{installDir}/langchain_community-0.3.2-py3-none-any.whl

!pip install -U vllm   --no-index --find-links=file:///kaggle/input/vllm-inference/





class CFG:

    # Use DeepEval framework for LLM Evaluation (only for Online, not for submission)
    DeepEval = False#True  # True must enable "Internet on" 
    USE_VLLM = False  # for inference
    USE_OVERTHINKING = False  # for deep seek new reasoning model, control and extend the thinking window provide  

    BASELINE = True # False
    RAG  = False   
    TRAIN = False 
    
    # LLM Config
    reportTo = "none"
    topK= 40
    topP = 1.0
    temperature = 0.6
    repetition_penalty= 1.05 #1.1
    maxOutToken = 1024

    # fine tuning Config
    maxLength = 1024 #token size
    maxExpoch = 2
    evalSteps = 20 
    learning_rate = 2e-4 #1e-4
    per_device_train_batch_size = 8#2#1
    per_device_eval_batch_size =  8 #2 #1

    warmup_steps= 5# 10
    gradient_accumulation_steps =2  #10
    maxTrainStep = 150 
    valDatasetSize = 20 # set validiation data size

    model1 = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b/2"
    model2 = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-14b/2"
    model3 = "/kaggle/input/qwen-qwq-32b/transformers/qwen-qwq-32b/1"


    refFile = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    sampleFile = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/sample_submission.csv"
    testFile = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv"
    
    

 


from dataclasses import dataclass
import torch
import torch.nn as nn
import os, time, json , gc
from IPython.display import display, Markdown, HTML


import pandas as pd
import polars as pl
import numpy as np

import torch
import kaggle_evaluation.aimo_2_inference_server

import asyncio

import ctypes


import transformers
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          TrainingArguments, BitsAndBytesConfig)

import pydantic

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

from vllm import LLM, SamplingParams # for vllm inference 


# Fine tuning
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

import asyncio


import kaggle_evaluation.aimo_2_inference_server



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# if torch.cuda.is_available():
#     print("GPUs are available!")
# else:
#     print("No GPUs found.")


if device.type =="cuda":
    !nvidia-smi


# if device.type =="cuda":
#     # Initialize distributed
#     rank = int(os.environ["RANK"])
#     device = torch.device(f"cuda:{rank}")
#     torch.distributed.init_process_group("nccl", device_id=device)


def clearMemory():
    for _ in range(5):
        torch.cuda.empty_cache()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        gc.collect()
        time.sleep(0.3)


clearMemory()


refDF = pd.read_csv(CFG.refFile)
refDF


refDF.describe()


sampleDF = pd.read_csv(CFG.sampleFile)
sampleDF


testDF = pd.read_csv(CFG.testFile)
testDF








# from vllm import LLM, SamplingParams

bnbConfig = BitsAndBytesConfig(
    load_in_4bit =True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)


if device.type =="cuda":
    model = AutoModelForCausalLM.from_pretrained(CFG.model3, 
                                                  device_map="auto",  
                                                 quantization_config= bnbConfig)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model3)


else:
    model = AutoModelForCausalLM.from_pretrained(CFG.model3, 
                                                  device_map="auto",
                                                  torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model3, use_fast=True)
    torch.set_num_threads(4)  # CPU accelerate

tokenizer.add_eos_token = True
tokenizer.padding_side= "right"



def delModel():
    global model, tokenizer
    del model
    del tokenizer
    clearMemory()


model


# tokenizer
# delModel()


clearMemory()


async def generateResponse(query, maxOutToken=CFG.maxOutToken,  topP=CFG.topP,
                          topK=CFG.topK, temperature=CFG.temperature):
    """
    Direct send message to LLM model, get response
    """
    global model, tokenizer
    startTime = time.time()
    inputIds = tokenizer(query, return_tensors="pt").to(device)
    response = model.generate(**inputIds,
                              do_sample=True,
                              top_p=topP,
                              top_k =topK,
                              temperature=temperature,
                              max_new_tokens =maxOutToken
                             )
    print(f"Time Taken : {time.time() - startTime}")
    # return tokenizer.decode(response[0][len(inputIds["input_ids"]):], skip_special_tokens = True)
    generatedIDs = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputIds.input_ids, response)
    ]
    # print(f"GeneratedIDs : {generatedIDs}")
    return tokenizer.batch_decode(generatedIDs, skip_special_tokens=True)[0]
    


def generateChatInstMsg(instruct, query):
    return [
            {
            "role" : "system",
             "content": instruct   
            },
           {
            "role" : "user",
            "content" : query   
           },
        ]


async def generateChatResponse(chatMsg, maxOutToken=CFG.maxOutToken,  topP=CFG.topP,
                          topK=CFG.topK, temperature=CFG.temperature):
    """
    send chat message to LLM
    """
    startTime =time.time()
    text = tokenizer.apply_chat_template(chatMsg, 
                                         tokenize=False,
                                        add_generation_prompt=True)
    inputIDs = tokenizer(text, return_tensors="pt").to(device)
    response = model.generate(**inputIDs,
                              do_sample=True,
                              top_p =topP,
                              top_k = topK,
                              temperature = temperature,
                              max_new_tokens= maxOutToken,
                              repetition_penalty = CFG.repetition_penalty)
    print(f"Time Taken : {time.time() - startTime}")
    generatedIDs = [
        output_ids[len(input_ids):] for input_ids , output_ids in zip(inputIDs.input_ids , response)
    ]
    return tokenizer.batch_decode(generatedIDs , skip_special_tokens=True)[0]
    





# ret =await generateResponse("What is Machine Learning?",    maxOutToken=512, topP=0.95,
#                           topK=10, temperature = 0.6)
# Markdown(ret)





# msg1 = generateChatInstMsg("You are a intelligent Chatbot response to answer user query", 
#                            "What is LLM model use case?")
# ret = await generateChatResponse(msg1, maxOutToken=1024, topP=0.95,
#                           topK=20, temperature = 0.6)
# Markdown(ret)


clearMemory()


from typing import Dict, List


# define output structure
class MathAnswer(BaseModel):
    question : str = Field(description="Describe the Mathematics question")
    reason:   str = Field(description = "Describe the Mathematics Solving step")
    anwser :  int = Field (description = "calculate final answer in numberic format")
    
    


parser1 = PydanticOutputParser(pydantic_object=MathAnswer)


print(parser1.get_format_instructions())








promptTemplate1 = ("""You are expert a mathematics, answer the Math question, step-by-step reasoning and thinking how accurate calucate the math answer relate to question.
Solve question step-by-step reasoning, thinking and how to use intermediate step to accurate calucate final answer.
###
question : {query}
                     
### 
answer : 
""")

promptTemplate2 = ("""You are expert a mathematics, answer the Math question, step-by-step reasoning and thinking how accurate calucate the math answer relate to question.
Solve question step-by-step reasoning, thinking and how to use intermediate step to accurate calucate final answer.
###
output format: {output}
###
question : {query}
                     
### 
""")


print(promptTemplate2)


instructionTemplate1 = ("Response to answer the Math question, step-by-step reasoning and thinking how accurate calucate the math answer relate to question.\n"
                        "does not repeat this instruction context in answer."
                        )

instructionTemplate2 = ("Response to answer the Math question, step-by-step reasoning and thinking how accurate calucate the math answer relate to question.\n"
                        "Output in JSON format"
                        )

instructionTemplate3 = ("Response to answer the Math question, step-by-step reasoning and thinking how accurate calucate the math answer relate to question.\n"
                        "Answer much be numberic.\n"
                        "Output in JSON format with key : problem, step-by-step-reason, answer"
                        )


instructionTemplate1







# prompt =prompt1.format(
#    query = 
    
# )


def extractAnswer(ret, jsonFormat= True):
    ans=None
    try:
        if jsonFormat:
            ans=ret["answer"]
            ans = ans % 1000 
        else:
            pass
    except:
        print("extract Answer Error")
    return ans        


  


async def testValidDataset(df , instruct=instructionTemplate1, chatMode = True, topP=CFG.topP,
          topK=CFG.topK,  temperature= CFG.temperature, 
          maxOutToken= CFG.maxOutToken, maxNumData=10):
    """
    for test Valid dataset
    """
    maxRetry = 3
    finalGen = []
    for idx in range (len(df)):
        print(f"Index : {idx}")
        rowData = df.iloc[idx]
        problem =  rowData["problem"]
        print(f"Problem : {problem}")
        ans = rowData["answer"]
        with torch.no_grad():
            if chatMode:
                tempChat = generateChatInstMsg(instruct, problem)
                ret = await generateChatResponse(tempChat, maxOutToken=maxOutToken,  topP=topP,
                          topK=topK, temperature=temperature)
            else:
                # prompt1 = PromptTemplate(
                # template=promptTemplate1,
                # input_variables=["query"],
                # partial_variables={"format_instructions": parser1.get_format_instructions()},
                # )
                # tempPrompt = promptTemplate1.format(
                #         query = problem)
                tempPrompt = promptTemplate2.format(
                    query =problem,
                    output= parser1.get_format_instructions()
                )
                print(f"TempPrompt : {tempPrompt}")
                ret = await  generateResponse(tempPrompt, maxOutToken=maxOutToken,  topP=topP,
                          topK=topK, temperature=temperature)

            print(f"LLM output :\n{ret}")
            print(f"Actual Answer : {ans}")
            print("*" * 30)

        finalGen.append(ret)
        if idx >= maxNumData -1:
            break
    
    return finalGen
    


async def infer(test, subDF, instruct=instructionTemplate1 , chatMode = True, topP=CFG.topP,
          topK=CFG.topK,  temperature= CFG.temperature, 
          maxOutToken= CFG.maxOutToken):
    finalGen = []
    maxRetry = 3
    for i , idx in enumerate(subDF["id"]):
        rowIdx  =  test.index[test["id"] == idx].tolist()[0] #find row Idx 
        problem  =  test.iloc[rowIdx]["problem"]
        print(problem)
        with torch.no_grad():
            if chatMode:
                tempChat = generateChatInstMsg(instruct, problem)
                ret = await generateChatResponse(tempChat, maxOutToken=maxOutToken,  topP=topP,
                          topK=topK, temperature=temperature)
            else:
                tempPrompt = promptTemplate1.format(
                        query = problem)
                ret = await  generateResponse(tempPrompt,maxOutToken=maxOutToken,  topP=topP,
                          topK=topK, temperature=temperature)

        finalGen.append(ret)
        

    return finalGen
                
    


def predict(id_: pl.DataFrame, question: pl.DataFrame) ->  pl.DataFrame | pd.DataFrame:
    """Make a prediction"""
    # Unpack values
    idx = id_.item(0)
    query = question.item(0)
    # Make a prediction
    prediction = 0  # model.predict(query)
    return pl.DataFrame({'id': idx, 'answer': 0})
    


ret = await testValidDataset(refDF, instruct=instructionTemplate1, chatMode = False, topP=0.5,
          topK=3,  temperature= 0.5,  maxOutToken= 4096, maxNumData=2)




Markdown(ret[0])


# parser1.invoke(ret[0])


ret2 = await infer(testDF, sampleDF, instruct=instructionTemplate3, chatMode =True, topP=0.5,
          topK=3,  temperature= 0.5, maxOutToken= 4096)


ret2


Markdown(ret2[2])


# parser1.invoke(output)



Markdown(ret2[0])


# pd.read_csv(
#     CFG.refFile
# ).drop('answer', axis=1).to_csv('reference.csv', index=False)


# aimo_infer_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

# if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
#     aimo_infer_server.serve()

# else:
#     aimo_infer_server.run_local_gateway(
#         (
#             # "reference.csv"
#             CFG.refFile
#         ))
    














# for k, v in refDF.items():
#     print(v)

# refDF.iloc[0]["answer"]
    

