installDir = "/kaggle/input/universal-llm-install-package2/V4" #"/kaggle/input/universal-llm-install-package2/V7"
!pip install transformers --no-index --no-deps --find-links=file://{installDir}/transformers-4.45.2-py3-none-any.whl
!pip install -U accelerate --no-index --no-deps --find-links=file://{installDir}/accelerate-1.0.1-py3-none-any.whl
!pip install -U trl --no-index --no-deps --find-links=file://{installDir}/trl-0.11.4-py3-none-any.whl
!pip install -U peft  --no-index --no-deps --find-links=file://{installDir}/peft-0.13.2-py3-none-any.whl
!pip install  bitsandbytes --no-index --no-deps --find-links=file://{installDir}/bitsandbytes-0.44.1-py3-none-manylinux_2_24_x86_64.whl


# 


from dataclasses import dataclass
import torch
import torch.nn as nn
import os, time , json, gc
from IPython.display import display, Markdown

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import transformers
from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                          AutoModelForSequenceClassification, TrainingArguments,
                          BitsAndBytesConfig)

from datasets import Dataset, DatasetDict

# fine tuning
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

import asyncio

# Metric
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, log_loss,
                             f1_score, accuracy_score, precision_score, recall_score)



class CFG:

    test = 0.2
    frac = 0.5 #0.3 #0.3 #0.3 #0.1 # Random resample data(reduce train data size )

    # LLM Config
    maxLength = 1024 #1600 #2048 #4096 # 2048 #6144# 4096 #5120 # 6144 #8192 # 3072  # 1024 #1600 #512     # token max lenght
    instTrain = False # prompt instruction Training
    reportTo = "none"
    topK = 10
    topP = 1.0
    temperature = 0.1
    maxEpoch= 0.05 #0.2 #1
    evalSteps = 50 #30 #40 #20 #40 #20  #20#50 #20 #50 #20 #50#20
    learning_rate = 2e-4#1e-4 #2e-4 #1e-4 #2e-4 # 1e-4
    per_device_train_batch_size = 16#8#6# 4 #6#8 #3#2 #4#2#8#4 #2#1
    per_device_eval_batch_size = 16 #8#6 #4 #6#8 #3 #2 #4#2 #8 #4 #2 #1 
    
    warmup_steps = 5 #10
    gradient_accumulation_steps = 2 #10
    maxTrainStep= 110 #120 #100#150 #60 #50 #120#100 #80 #150 #200 #60 #70 #50 #40 #50#70 #100#250#400 #100 #350 #200
    valDatasetSize = 300 #150 #300  #100 #150 #200#800 #1000 #400 #300 #200#300 # set validiation data size

    
    # model1 = "/kaggle/input/qwen2.5/transformers/1.5b-instruct/1"
    model2 = "/kaggle/input/qwen2.5/transformers/3b-instruct/1"
    model3 = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"

    trainFile = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet"
    testFile =  "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet"
    sampleFile = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv"
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


torch.cuda.is_bf16_supported()


def clearMemory():
    for _ in range(5):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)
        


clearMemory()


def printSeriesUniqueVal(series):
    print(f"{series.unique()}")


def printAllColumnsValue(df, showAll=True):
    for col in df.columns:
        if showAll:
            print(f"{col} : {df[col].unique()}") # print unique value
        else:
            if df[col].dtype == "object": #only print catergory type column
                print(f"{col} : {df[col].unique()}")


trainDF = pd.read_parquet(CFG.trainFile)
trainDF


testDF = pd.read_parquet(CFG.testFile)
testDF


testDF.columns


submit = pd.read_csv(CFG.sampleFile)
submit


trainDF["language"].value_counts()[:50] # top 50 language


trainDF["winner"].value_counts().plot(kind="bar", title="Winner Distribution");


trainDF.columns


listModelA = trainDF["model_a"].unique()
listModelB = trainDF["model_b"].unique()
len(listModelA),  len(listModelB) # 60 LLM model


sorted(listModelA)


sorted(listModelB)


trainDF.isnull().sum() # no null value


id2Label = {0: "model_a", 1 : "model_b"}
label2id = {"model_a" : 0 , "model_b": 1}


label2id


id2Label


bnbConfig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# if load in 4bit not work, use as below bnbconfig
# bnbConfig = BitsAndBytesConfig(
#     load_in_8bit=True,
# )


# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "<s>"
# DEFAULT_UNK_TOKEN = "<unk>"

if device.type == "cuda":
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model2, #CFG.model3,
        num_labels= 2,
        id2label =id2Label,
        label2id = label2id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config= bnbConfig,
        trust_remote_code=False,
        problem_type="single_label_classification", 
    )

else: 
     model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model2, #CFG.model3,
        num_labels= 2,
        id2label =id2Label,
        label2id = label2id,
        device_map="auto",
        trust_remote_code=False,
        problem_type="single_label_classification", 
    )
tokenizer = AutoTokenizer.from_pretrained(CFG.model2)

# tokenizer.add_special_tokens(
#         {
#             "pad_token": DEFAULT_PAD_TOKEN,
#             "eos_token": DEFAULT_EOS_TOKEN,
#             "bos_token": DEFAULT_BOS_TOKEN,
#             "unk_token": DEFAULT_UNK_TOKEN,
#         }
# )
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_eos_token = True  
tokenizer.padding_side = "right"


!nvidia-smi  


model


trainInstruct ="""###You are intelligent machine, 
it is a Text Classification task to predict the human preference response given input prompt and 2 models response_a and response_b respectively.
Provided the label indicate which response more user preference###"""


trainDF


trainDF.columns


trainDF["label"] = trainDF["winner"].map(label2id) # map model_a=0, model_b =1


trainDF


trainDF.isnull().sum()


if CFG.instTrain:
    # trainDF["LLM Content"] = ( trainInstruct + "\nprompt : " + trainDF["prompt"] +
    #                       "\nresponse_a : " + trainDF["response_a"]  +
    #                       "\nresponse_b : " + trainDF["response_b"])
    trainDF["LLM Content"] = ("<prompt>: " + trainDF["prompt"] +
                          "\n\n<response_a>: " + trainDF["response_a"]  +
                          "\n\n<response_b>: " + trainDF["response_b"])
else:
     # trainDF["LLM Content"] = ("prompt : " + trainDF["prompt"] +
     #                      "\nresponse_a : " + trainDF["response_a"]  +
     #                      "\nresponse_b : " + trainDF["response_b"])
     trainDF["LLM Content"] = ("<prompt>: " + trainDF["prompt"] +
                          "\n\n<response_a>: " + trainDF["response_a"]  +
                          "\n\n<response_b>: " + trainDF["response_b"])


# testDF["LLM Content"] = ( "prompt" + testDF["prompt"] +
#                        "\nresponse_a : " + testDF["response_a"]  +
#                           "\nresponse_b : " + testDF["response_b"])
testDF["LLM Content"] = ( "<prompt>: " + testDF["prompt"] +
                       "\n\n<response_a>: " + testDF["response_a"]  +
                          "\n\n<response_b>: " + testDF["response_b"])
                    






Markdown(trainDF["LLM Content"][0])


len(trainDF["LLM Content"][2])


trainDF["Lenght"]=trainDF["LLM Content"].str.len() # counting LLM content length



 trainDF["Lenght"].plot(kind="hist", bins=200);


trainDF["Lenght"].value_counts()[:50]


# largeLen = trainDF[trainDF["Lenght"] > 6122]
# largeLen





#frac=0.1 # 0.25
newtrainDF = trainDF.sample(frac=CFG.frac)
newtrainDF = newtrainDF.reset_index(drop=True)


newtrainDF["label"].value_counts()


len(newtrainDF), round(len(newtrainDF) *0.8)


maxTrainData = round(len(newtrainDF) *0.95)
maxTrainData


newtrainDF = newtrainDF[newtrainDF["Lenght"] <CFG.maxLength]
newtrainDF =newtrainDF.reset_index(drop=True)


newtrainDF


newtrainDF["label"].value_counts()


# 1292/4834


del trainDF


totalTrainSize = len(newtrainDF)
totalTrainSize


setMaxTrainData= totalTrainSize- CFG.valDatasetSize
setMaxTrainData 


tempTrainDF = newtrainDF[:setMaxTrainData]
tempValDF = newtrainDF[setMaxTrainData:]


trainDataset = Dataset.from_pandas(tempTrainDF, split="train")
evalDataset = Dataset.from_pandas(tempValDF, split="test")
subDataset = Dataset.from_pandas(testDF, split="test")


trainDataset


evalDataset


subDataset


submitDict = DatasetDict({
        "test": subDataset
})


submitDict


datasetDict = DatasetDict( {
        "train" : trainDataset,
        "test"  : evalDataset
})


datasetDict


del tempTrainDF 
del tempValDF


# tokenizer


# convert Tokenizer function, take care max_lenght size match to LLM content size
def tokenizerFunc(sample):
    return tokenizer(sample["LLM Content"], max_length=CFG.maxLength, padding=True, truncation=True)


# apply tokenizer function
datasetDict = datasetDict.map(tokenizerFunc, batched=True)


datasetDict


submitDict.map(tokenizerFunc, batched=True)


datasetDict = datasetDict.rename_column("label", "labels") 
#  Rename the label column to labels because the model expects the argument to be named labels





lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        task_type="SEQ_CLS",  # for sequence classification
        bias= "none",
        # target_modules=["q_proj", "k_proj", "v_proj"],
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout= 0.1, #0.08,#0.05, #0.1, #0.05,
)


estimTrainStep = round(CFG.maxEpoch * len(datasetDict["train"]))
estimTrainStep #esimate number of setp for whole epoch


model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


model.print_trainable_parameters()


# extract train cloum format
def formatFuc(sample):
    text = f"{sample['LLM Content']}"
    return [text]


# accList= []
# f1List= []
# recallList = []
# preciseList = []
def compute_metric(pred):
    logits, labels = pred
    predication = np.argmax(logits, axis= -1) # for multi-class
    acc = accuracy_score(labels, predication)
    f1  = f1_score(labels, predication, average="weighted", zero_division=1)
    recall = recall_score(labels, predication, average='weighted', zero_division=1)
    precision = precision_score(labels, predication, average='weighted', zero_division=1)
    # accList.append(acc)
    # f1List.append(f1)
    # recallList.append(recall)
    # preciseList.append(precision)
    return {"accuracy": acc, "recall":  recall, "precision": precision, "f1-score": f1}


trainArg = TrainingArguments(
    output_dir = "kaggle/working",
    per_device_train_batch_size = CFG.per_device_train_batch_size,
    per_device_eval_batch_size = CFG.per_device_eval_batch_size, 
    gradient_accumulation_steps = CFG.gradient_accumulation_steps, #10,
    eval_strategy="steps",
    save_strategy="steps",
    warmup_steps=  CFG.warmup_steps, #10,
    max_steps= CFG.maxTrainStep, 
    learning_rate= CFG.learning_rate,
    fp16=True,
    # bf16=True,
    logging_steps=CFG.evalSteps,
    eval_steps = CFG.evalSteps,
    optim= "paged_adamw_8bit", #"adamw_8bit", 
    report_to= CFG.reportTo,
    do_eval=True,                # Perform evaluation at the end of training
    weight_decay=0.01,
    # load_best_model_at_end=True
    
)


# trainArg


from transformers import DataCollatorWithPadding


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = SFTTrainer(
        model=model,
        train_dataset=datasetDict["train"],
        eval_dataset = datasetDict["test"],
        args=trainArg,
        peft_config= lora_config,
        data_collator=data_collator,
        max_seq_length= CFG.maxLength,
        compute_metrics= compute_metric
)


# Fix Batch size >1 , not define pad token id issues
eos_token_id=tokenizer.eos_token_id
model.config.use_cache=False
model.config.pad_token_id = tokenizer.pad_token_id


clearMemory()


# model.config.pad_token_id


%%time
trainer.train()


model.save_pretrained("multilingual-chatbot-arena-llm-Fine-tune")


trainHistoryDF = pd.DataFrame(trainer.state.log_history)
trainHistoryDF


import matplotlib.pyplot as plt
trainloss = trainHistoryDF[~trainHistoryDF["loss"].isnull()]
valloss = trainHistoryDF[~trainHistoryDF["eval_loss"].isnull()]
plt.plot(trainloss["loss"], label="Train")
plt.plot(valloss["eval_loss"], label="Val")
plt.title("LLM loss")
plt.legend()
plt.show()


trainloss = trainHistoryDF[~trainHistoryDF["eval_accuracy"].isnull()]
plt.plot(trainloss["eval_accuracy"], label="Val Accuracy")
plt.title("Val Accuracy")
plt.legend()
plt.show()


trainloss = trainHistoryDF[~trainHistoryDF["eval_recall"].isnull()]
plt.plot(trainloss["eval_recall"], label="Val Recall")
plt.title("Val Recall")
plt.legend()
plt.show()


trainloss = trainHistoryDF[~trainHistoryDF["eval_precision"].isnull()]
plt.plot(trainloss["eval_precision"], label="Val Precision")
plt.title("Val Precision")
plt.legend()
plt.show()


trainloss = trainHistoryDF[~trainHistoryDF["eval_f1-score"].isnull()]
plt.plot(trainloss["eval_f1-score"], label="Val F1-Score")
plt.title("Val F1-Score")
plt.legend()
plt.show()


clearMemory()


model.config.use_cache=True # for inference


 def testValidDataset(ds , maxNumData=10):
     for i , data in enumerate(ds["test"]):
         print(f"data {i}:")
         # newPrompt = "prompt : " + data["prompt"] + \
         #              "\nresponse_a : " + data["response_a"]  + \
         #              "\nresponse_b : " + data["response_b"]
         newPrompt = "<prompt>: " + data["prompt"] + \
                      "\n\n<response_a>: " + data["response_a"]  + \
                      "\n\n<response_b>: " + data["response_b"]
         with torch.no_grad():
             inputIds = tokenizer(newPrompt, return_tensors="pt").to(device)
             logits = model(**inputIds).logits
             probabilities = nn.functional.softmax(logits, dim=-1) # get probilitity
             classID =logits.argmax().item()
             classTxt = model.config.id2label[classID]
             print(f"Query : {newPrompt}\n\rPredict Class ID : {classID}\n\rPredict Winner Name: {classTxt}\n\rActual Class ID: {data['labels']}")
             print("-"*50)
         if i >= maxNumData -1:
            break
             


testValidDataset(datasetDict, 3)


submit["id"]


submitDict['test']


def inferDF(ds, sub):
    """
    input : ds = submit Test dataset, sub = sample submission file
    """
    finalpredict = []
    for i, subId in enumerate(sub["id"]):
        print(f"Sub id : {subId}")
        rowIdx = ds["test"]["id"].index(subId)
        # print(f"row idx : {rowIdx}")
        rowData = ds["test"][rowIdx]
        # newPrompt = "prompt : " + rowData["prompt"] + \
        #               "\nresponse_a : " + rowData["response_a"]  + \
        #               "\nresponse_b : " + rowData["response_b"]
        newPrompt = "<prompt>: " + rowData["prompt"] + \
                      "\n\n<response_a>: " + rowData["response_a"]  + \
                      "\n\n<response_b>: " + rowData["response_b"]
        
        with torch.no_grad():
            inputIds = tokenizer(newPrompt, return_tensors="pt").to(device)
            logits = model(**inputIds).logits
            probabilities = nn.functional.softmax(logits, dim=-1) # get probilitity
            classID =logits.argmax().item()
            classTxt = model.config.id2label[classID]
            # print(f"Query : {newPrompt}\n\rPredict Class ID : {classID}\n\rPredict Winner Name: {classTxt}")
            # print("-"*50)
            finalpredict.append(classTxt)

    return finalpredict
            
            


finalPredict = inferDF(submitDict, submit)


finalPredict


submit["winner"] = finalPredict


submit


submit.to_csv("submission.csv", index=False)


pd.read_csv("submission.csv")







