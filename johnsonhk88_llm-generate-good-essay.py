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

# install deepeval require dependent framework
# !pip install -U openai   --no-index --no-deps --find-links=file://{installDir2}/openai-1.60.2-py3-none-any.whl
# !pip install -U deepeval --no-index --no-deps --find-links=file://{installDir2}/deepeval-2.2.6-py3-none-any.whl
# !pip install -U portalocker --no-index --no-deps --find-links=file://{installDir2}/portalocker-3.1.1-py3-none-any.whl


class CFG:

    
    # Use DeepEval framework for LLM Evaluation (only for Online, not for submission)
    DeepEval = False#True  # True must enable "Internet on" 

    # LLM Config 
    reportTo ="none"
    topK = 40
    topP = 1.0
    temperature = 0.1 #0.5
    repetition_penalty = 1.05 # 1.1
    maxOutToken = 150#180 #100
    

    # Fine tuning Config
    
    maxLength = 768 #1024 
    reportTo = "none"
    maxEpoch = 2
    evalSteps = 20 
    learning_rate = 2e-4 #1e-4
    per_device_train_batch_size = 8#6 #8
    per_device_eval_batch_size =  8#6 #8  

    warmup_steps= 5# 10
    gradient_accumulation_steps =2  #10
    maxTrainStep = 250 #150 
    valDatasetSize = 20 # set validiation data size
   

    
    
    

    trainFile = "/kaggle/input/synthetic-essay-topics/synthetic_essays.csv"
    
    
    sampleFile = "/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv"
    testFile = "/kaggle/input/llms-you-cant-please-them-all/test.csv"
    model1 = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"
    model2 = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/1"
    model3 = "/kaggle/input/deepseek-r1-distill-qwen-1.5b/transformers/deepseek-r1-distill-qwen-1.5b/1"
    model4 = "/kaggle/input/deepseek-r1-distill-qwen-1.5b/transformers/deepseek-r1-distill-qwen-7b/1"
    model5 = "/kaggle/input/deepseek-r1-distill-qwen-1.5b/transformers/deepscaler-1.5b-preview/1"


if CFG.DeepEval: # requirement online
    !pip install  -U deepeval


import os,time, json, gc
from IPython.display import display, Markdown

from typing import List
from pydantic import BaseModel, Field 


import torch
import torch.nn as nn
import transformers
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                         TrainingArguments, BitsAndBytesConfig)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#
from datasets import Dataset, DatasetDict

#supervised fine tuning
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

import asyncio

# Metric
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, log_loss,
                            f1_score, accuracy_score, precision_score, recall_score)

from langchain_core.output_parsers import StrOutputParser
   


if CFG.DeepEval:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval
    # from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric
    # from deepeval.benchmarks import MMLU, HellaSwag, HumanEval,TruthfulQA
    # from deepeval.benchmarks.tasks import HumanEvalTask, HellaSwagTask, MMLUTask, TruthfulQATask
    # from deepeval.benchmarks.modes import TruthfulQAMode 


testDF = pd.read_csv(CFG.testFile)
testDF


testDF["topic"][2]


sub = pd.read_csv(CFG.sampleFile)
sub


trainDF = pd.read_csv(CFG.trainFile, on_bad_lines='skip')
trainDF


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def clearMemory():
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)


clearMemory()


# Random Resample Dataset
trainDF = trainDF.sample(frac=1.0 , replace=True)
trainDF = trainDF.reset_index(drop=True)


trainDF


trainDF["LLM Context"] = "topic :" + trainDF["Topic"] + \
                         "\nessay : " + trainDF["Essay"]
                            


trainDF


trainDF["Length"]= trainDF["LLM Context"].str.len()


trainDF["Length"].describe()


len(trainDF) , round(len(trainDF) * 0.85)


totalTrainSize= len(trainDF)
totalTrainSize


setMaxTrainData = totalTrainSize - CFG.valDatasetSize
setMaxTrainData


tempTrainDF = trainDF[:setMaxTrainData]
tempValDF = trainDF[setMaxTrainData:]


len(tempValDF)


trainDataset = Dataset.from_pandas(tempTrainDF, split="train")
valDataset = Dataset.from_pandas(tempValDF, split="test")


trainDataset


valDataset


datasetDict = DatasetDict({
        "train" : trainDataset,
        "test" : valDataset
})


datasetDict


del tempTrainDF
del tempValDF


clearMemory()








# select Model
selModel = CFG.model5 #CFG.model3 # CFG.model4


bnbConfig = BitsAndBytesConfig(
    load_in_4bit =True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16,
    bnb_4bit_use_double_quant=True
    
)
bnbConfig


if device.type =="cuda": 
    model = AutoModelForCausalLM.from_pretrained(
            selModel, 
            quantization_config = bnbConfig,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False)
    
else:
     model = AutoModelForCausalLM.from_pretrained(
            selModel,
            # quantization_config = bnbConfig,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=False)

tokenizer = AutoTokenizer.from_pretrained(CFG.model3)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True  
tokenizer.padding_side = "right"


eos_token_id=tokenizer.eos_token_id
model.config.use_cache=False # for training 
model.config.pad_token_id = tokenizer.pad_token_id


model


# Convert LLM context to tokenizer
def tokenizerFun(sample):
    return tokenizer(sample["LLM Context"] , max_length=CFG.maxLength, padding=True, truncation=True)


datasetDict  = datasetDict.map(tokenizerFun,  batched=True) # generate tokenize value


datasetDict


lora_config = LoraConfig(
    r=16,
        lora_alpha=32,
        task_type="CAUSAL_LM",  # for generative  task
        bias= "none",
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout= 0.05, #0.1, #0.05,
)


model = get_peft_model(model, lora_config)


model.print_trainable_parameters()


estimTrainStep = round(CFG.maxEpoch * len(datasetDict["train"]))
estimTrainStep #esimate number of setp for whole epoch


# extract training column format
def formatFuct(sample):
    text = f"{sample['LLM Context']}"
    return [text]


trainArg = SFTConfig(
    output_dir= "kaggle/working",
    max_seq_length= CFG.maxLength,
    per_device_train_batch_size =CFG.per_device_train_batch_size,
    # per_device_eval_batch_size = CFG.per_device_eval_batch_size,
    gradient_accumulation_steps = CFG.gradient_accumulation_steps,
    # eval_strategy= "steps",
    save_strategy= "steps",
    warmup_steps=CFG.warmup_steps,
    max_steps = CFG.maxTrainStep,
    learning_rate=CFG.learning_rate,
    fp16=True,
    logging_steps= CFG.evalSteps,
    # eval_steps = CFG.evalSteps,
    optim="paged_adamw_8bit",
    report_to = CFG.reportTo,
    # do_eval=True,                # Perform evaluation at the end of training
    # weight_decay=0.01,
    
)


from transformers import DataCollatorWithPadding


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer= SFTTrainer(
    model=model,
    train_dataset=datasetDict["train"],
    # eval_dataset = datasetDict["test"],
    args=trainArg,
    peft_config = lora_config,
    formatting_func=formatFuct,
    # data_collator = data_collator, 
    
    
)


clearMemory()


%%time
trainer.train()


model.save_pretrained("LLM-generate-Good-essay-llm-Fine-tune")


trainHistoryDF = pd.DataFrame(trainer.state.log_history)
trainHistoryDF


trainloss = trainHistoryDF[~trainHistoryDF["loss"].isnull()]
plt.plot(trainloss["loss"], label="Train")
plt.title("Training Loss")
plt.legend()
plt.show()





model.config.use_cache=True # for inference 


async def generateResponse(query, maxOutToken=CFG.maxOutToken, topP=CFG.topP,
                          topK=CFG.topK, temperature = CFG.temperature):
    """
    Direct send message to LLM 
    """
    startTime = time.time()
    inputIDs = tokenizer(query, return_tensors="pt").to(device)
    response = model.generate(**inputIDs, 
                             do_sample=True,  #enable for Temperature 
                             top_p= topP,
                             top_k = topK,
                             temperature = temperature,
                             max_new_tokens=maxOutToken,
                             repetition_penalty= CFG.repetition_penalty)
    print(f"Time Taken : {time.time() - startTime}")
    # return tokenizer.decode(response[0][len(inputIDs["input_ids"]):], skip_special_tokens=True)
    generatedIDs = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputIDs.input_ids, response)
    ]
    # print(f"GeneratedIDs : {generatedIDs}")
    return tokenizer.batch_decode(generatedIDs, skip_special_tokens=True)[0]


def generateChatInstMsg(instruct, query):
    return   [
            {
            "role": "system",
            "content": instruct,
            },
            {"role": "user", 
             "content": query},
        ]


async def generateChatResponse(chatMsg ,maxOutToken=CFG.maxOutToken, topP=CFG.topP,
                          topK=CFG.topK, temperature = CFG.temperature):
    """
    send chat message to LLM
    """
    startTime = time.time()
    text = tokenizer.apply_chat_template(chatMsg, 
                                         tokenize=False, 
                                         add_generation_prompt=True)
    inputIDs = tokenizer(text, return_tensors="pt").to(device)
    response = model.generate(**inputIDs, 
                             do_sample=True,  #enable for Temperature 
                             top_p= topP,
                             top_k = topK,
                             temperature = temperature,
                             max_new_tokens=maxOutToken,
                             repetition_penalty= CFG.repetition_penalty)
    print(f"Time Taken : {time.time() - startTime}")
    generatedIDs = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputIDs.input_ids, response)
    ]
    # print(f"GeneratedIDs : {generatedIDs}")
    return tokenizer.batch_decode(generatedIDs, skip_special_tokens=True)[0]
    # return tokenizer.decode(response[0][len(inputIDs["input_ids"]):], skip_special_tokens=True)
    


# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
# tokenized_chat
# # print(tokenizer.decode(tokenized_chat[0]))



queryTest1= "What is Deep Learning and LLM Model?"


ret = await generateResponse(queryTest1)
Markdown(ret)


len(ret)


clearMemory()





msg1 = generateChatInstMsg("You are a intelligent Chatbot response to answer user query", 
                           "What is LLM model use case?")


ret = await generateChatResponse(msg1)
Markdown(ret)


promptTemplate1 = """You Are expert a eassy writer,response to Generate the Human like style good quality of English essay by User given topic as below\n
                     generate 100-words eassy, step-by-step reasoning and thinking how generate godd quality the eassy relate to topic like human being
                     ###
                     topic : {title}
                     
                     ### 
                     generate : 
                  """

promptTemplate2 = """You are expert a eassy writer, response to Generate the Human like style good quality of English essay by User given topic as below\n
                     generate 100-words eassy, step-by-step reasoning and thinking how generate good quality the eassy relate to topic like human being, does not explaining how to generate eassy.\n
                     Make ensure generate essay is completed sentence.
                     ###
                     topic : {title}
                     
                     ### 
                     generate : 
                  """

instructionTemplate1 = "You Are intelligent Chatbot, response to Generate the Human like style Good Quality of English essay by User given topic"
instructionTemplate2 = """You Are expert eassy writer, response to Generate the Human like style Good Quality of English essay by User given topic
                          Output Maximum less than 100 words.
                        """

instructionTemplate3 = ("Generate the Human like style Good Quality of English essay by User given topic, output maximum 100 words essay, step-by-step reasoning and thinking how generate good quality the eassy relate to topic like human being.\n"
                        "Does not explain the task and how to genernate eassy and does not repeat this instruction context in generation essay. Must generated a completed sentence.")

instructionTemplate4 = ("Generate the Human like Good Quality of English essay by User given topic, step-by-step reasoning and thinking how generate good quality the eassy relate to topic.\n"
                        "Does not explain the task and how to genernate eassy and does not repeat this instruction context in generation essay.\nMust generated a completed sentence.\n\n"
                        "consistent Quality: Create essays that start strongly but degrade in quality towards the end, or vice versa.\n"
                        "Varying Complexity: Alternate between simple and complex sentence structures within the same essay.\n"
                         "Englist Grammar and Spelling: Maintain correct grammar and spelling to avoid penalties.\n"
                         "Natural Language: Ensure the generated text sounds natural and human-written.\n\n"
                          "Avoid Repetition: Implement a mechanism to track and minimize the similarity between generated essays.\n"
                         "Topic Diversity: Generate essays covering a wide range of topics.\n"
                         "Each topic must be generate around 100 words, and last sentence must be completed sentence.\n"
                          "Randomly Ouput different Written Style")

instructionTemplate5 = ("Generate the Human like Good Quality of English essay by User given topic, step-by-step reasoning and thinking how generate good quality the eassy relate to topic.\n"
                        "Does not explain the task and how to genernate eassy and does not repeat this instruction context in generation essay.\nMust generated a completed sentence.\n\n"
                        "consistent Quality: Create essays that start strongly but degrade in quality towards the end, or vice versa.\n"
                        "Varying Complexity: Alternate between simple and complex sentence structures within the same essay.\n"
                         "Englist Grammar and Spelling: Maintain correct grammar and spelling to avoid penalties.\n"
                         "Natural Language: Ensure the generated text sounds natural and human-written.\n\n"
                          "Avoid Repetition: Implement a mechanism to track and minimize the similarity between generated essays.\n"
                         "Topic Diversity: Generate essays covering a wide range of topics.\n"
                         "Each topic must be generate around 100 words, and last sentence must be completed sentence.\n"
                         "Randomly Output different Written Style as below:\n\n"
                         "Persuasive: Aims to convince the reader of a certain viewpoint using arguments and evidence.\n"
                         "Narrative: Focuses on storytelling, including characters, conflict, and settings.\n"
                         "Expository: Conveys information and explains a topic factually, avoiding personal opinions.\n"
                         "Descriptive: Uses vivid language and sensory details to create a picture in the reader's mind.\n"
                         "Analytical: Examines and evaluates a topic or argument in a detailed and structured way.\n"
                         "Technical: Conveys complex information in a clear and precise manner, often related to science, engineering, or technology.\n"
                          "Journalistic: Presents facts in an objective and unbiased manner, commonly used in news reporting.\n"
                         "Creative: Focuses on originality, imagination, and artistic expression.\n"
                         "Lyrical: Characterized by an expressive, musical style, often conveying strong emotions.\n"
                         "Minimalist: Emphasizes simplicity and conciseness in language and structure.\n"
                         "Stream-of-consciousness: Presents thoughts and feelings in a continuous, unstructured flow.\n"
                         "Beat Literature-inspired: Focuses on inner states, evocative environmental descriptions, simple language, repetition, long sentences, and the rhythm of language.\n"
                         "Early 20th Century Modernism: Mixes long and short sentences, details inner states, and focuses on dialogue interludes.\n"
                          "Melodic: Employs structured forms like period or sentence form to create a musical quality.\n"
                          "Terse & Dense: A writing style that is direct, compact, and information-rich.")


print(instructionTemplate5)


len(instructionTemplate5)


def customOutPraser(text):
    """
    custom filter unwant content
    """
    out = text.strip()
    out = out.replace("<think>", "")
    out = out.replace("</think>", "")
    out = out.replace("*/", "")
    
    return out


async def testEvalData(ds, instruct=instructionTemplate3,  chatMode=True, topP=CFG.topP,
                        topK=CFG.topK,  temperature= CFG.temperature, maxOutToken= CFG.maxOutToken,
                         maxTestItem=5):
    maxRetry = 3
    for i, data in enumerate(ds["test"]):
            topic = data["Topic"]
            numWords = 0
            reRetryCnt= 0
            while numWords < 90:
                if chatMode:
                    #generate Chat  Prompt
                    tempChat = generateChatInstMsg(instruct, topic)
                    # print(tempChat)
                    ret = await generateChatResponse(tempChat, maxOutToken=maxOutToken, topP=topP,
                          topK=topK, temperature =temperature)
                    ret = customOutPraser(ret)
                    numWords = len(ret.split())
                else:
                    # generate m
                    tempPrompt = promptTemplate2.format(
                        title = topic
                    )
                    ret = await generateResponse(tempPrompt, maxOutToken=maxOutToken, topP=topP,
                          topK=topK, temperature = temperature)
                    ret = customOutPraser(ret)
                    numWords = len(ret.split())
                reRetryCnt +=1 
                if reRetryCnt >= maxRetry: #check retry 
                    break
            print(f"Topic : {topic}\n\rTrain Essay: {data['Essay']}\n\n\rGenerate : {ret}\n\r")
            print("-"*50)
            if i>= maxTestItem:
                break
            
              
    


await testEvalData(datasetDict, instruct =instructionTemplate4,  chatMode=True, topP=0.9,
                          topK=120, temperature=0.6, maxTestItem=3)


async def infer(test, subDF, instruct= instructionTemplate3, chatMode = True, topP=CFG.topP,
                          topK=CFG.topK,  temperature= CFG.temperature, maxOutToken= CFG.maxOutToken):
    finalGen = []
    maxRetry = 3
    for i, idx in enumerate(subDF["id"]):
        # print(idx)
        rowIdx= test.index[test["id"] == idx].tolist()[0] # get index in testDF
        # print(rowIdx)
        # print(test.iloc[rowIdx]["topic"])
        topic = test.iloc[rowIdx]["topic"]
        numWords = 0
        reRetryCnt= 0
        while numWords < 90:
            if chatMode:
                #generate Chat  Prompt
                tempChat = generateChatInstMsg(instruct, topic)
                # print(tempChat)
                ret = await generateChatResponse(tempChat, maxOutToken=maxOutToken, topP=topP,
                          topK=topK, temperature =temperature)
                ret = customOutPraser(ret)
                numWords = len(ret.split())
            else:
                # generate m
                tempPrompt = promptTemplate2.format(
                        title = topic
                )
                ret = await generateResponse(tempPrompt, maxOutToken=maxOutToken, topP=topP,
                          topK=topK, temperature = temperature)
                ret = customOutPraser(ret)
                numWords = len(ret.split())
            reRetryCnt +=1 
            if reRetryCnt >= maxRetry: #check retry 
                break
        
        finalGen.append(ret)
    return finalGen



predict = await infer(testDF, sub, instruct =instructionTemplate4,  chatMode=True, topP=0.9,
                          topK=120, temperature=0.6)


len(predict[0].split()) # count words 



print(predict[1])


sub["essay"] = predict


sub


sub.to_csv("submission.csv", index=False)


pd.read_csv("submission.csv")







