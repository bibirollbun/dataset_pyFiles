# import kagglehub

DEBUG = False#True

if DEBUG is True:
    # !pip install --upgrade datasets accelerate  # add datasets, accelerate , bitsndbytes
    !pip install -U peft==0.14.0
    !pip install -U trl==0.11.4
    # !pip install --upgrade peft trl
    # !pip install -U datasets
    !pip install -U accelerate==1.0.1
    !pip install transformers==4.45.2
    !pip install langchain==0.3.27
    !pip install -U langchain-core==0.3.74
    !pip install langchain-text-splitters==0.3.9
    !pip install langchain-community==0.3.27
    # !pip install -U deepeval==3.3.9
    # !pip install protobuf==4.25.8 #4.49.0     # 6.30.2      #==3.20.3 #4.49.0
    # !pip install triton==3.0.0   # avoid load LLM error
    !pip install bitsandbytes==0.46.1
    # !pip install unsloth==2025.9.9
    !pip install unsloth==2025.9.1

else:
    # package = kagglehub.package_import("/johnsonhk88/universal-llm-install-package2")
    # installDir = "/kaggle/input/universal-llm-install-package2/V7" #"/kaggle/input/universal-llm-install-package2/V7"
    installDir = "/kaggle/input/universal-llm-install-package2/V9"
    # installDir2 ="/kaggle/input/deepeval-open-source-llm-evaluation-framework"
    !pip install transformers --no-index --no-deps --find-links=file://{installDir}/transformers-4.45.2-py3-none-any.whl
    !pip install -U accelerate --no-index --no-deps --find-links=file://{installDir}/accelerate-1.0.1-py3-none-any.whl
    !pip install -U trl --no-index --no-deps --find-links=file://{installDir}/trl-0.11.4-py3-none-any.whl
    !pip install -U peft --no-index --no-deps --find-links=file://{installDir}/peft-0.14.0-py3-none-any.whl
    !pip install  bitsandbytes --no-index --no-deps --find-links=file://{installDir}/bitsandbytes-0.46.1-py3-none-manylinux_2_24_x86_64.whl
    
    !pip install -U langchain --no-index  --no-deps --find-links=file://{installDir}/langchain-0.3.3-py3-none-any.whl
    !pip install -U langchain_core --no-index  --no-deps --find-links=file://{installDir}/langchain_core-0.3.12-py3-none-any.whl
    !pip install -U langchain_text_splitters  --no-index  --no-deps  --find-links=file://{installDir}/langchain_text_splitters-0.3.0-py3-none-any.whl
    !pip install -U langchain_community  --no-index  --no-deps   --find-links=file://{installDir}/langchain_community-0.3.2-py3-none-any.whl
    # !pip install -U triton  --no-index --no-deps  --find-links=file://{installDir}/triton-3.4.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
    # !pip install -U protobuf  --no-index  --no-deps  --find-links=file://{installDir}/protobuf-4.25.8-cp37-abi3-manylinux2014_x86_64.whl  
    
    # !pip install -U vllm   --no-index --find-links=file:///kaggle/input/vllm-inference/




# !pip install triton==3.0.0
# !pip install triton==3.3.0


# !pip install deepspeed==0.14.5


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os, time , gc , json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import transformers





# !pip install --upgrade imbalanced-learn
# !pip install --upgrade scikit-learn


# !pip install trl==0.15.0
# from imblearn.over_sampling import SMOTE
# from collections import Counte


import trl
trl.__version__


import torch
from sklearn.model_selection import train_test_split

from transformers import (AutoTokenizer, 
                          BitsAndBytesConfig, 
                          AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                         TrainingArguments)

from datasets import Dataset, DatasetDict, load_dataset


# Fine tuning 
from trl import SFTTrainer, SFTConfig
from peft import (LoraConfig, 
                    PeftModel, 
                    get_peft_model)
                    # prepare_model_for_kbit_training)  #prepare_model_for_int8_training deprecated 

from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, log_loss,
                             f1_score, accuracy_score, precision_score, recall_score)




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class CFG:
    DeepEval = False #True  # True must enable "Internet on" 
    SEED = 42

     # LLM Config 
    reportTo ="none"
    topK = 40
    topP = 1.0
    temperature = 0.1 #0.5
    repetition_penalty = 1.05 # 1.1
    maxOutToken = 300 #150#180 #100
    test_size = 0.003 #0.002 #0.005 #0.005 #0.1 #0.01 #0.005 #0.01


    # Fine tuning Config
    maxLength = 256 #384 #512 #256 #368 #512 #1024 
    reportTo = "none"
    maxEpoch = 0.014 #0.013 #0.005 #0.01 #0.005 #0.5 #0.05 #0.02 #0.05  #0.1 #0.5 #1 #2
    evalSteps = 20 #50 #30 #40 #20 
    learning_rate = 9e-4 #8e-4 #5e-4 #2e-5 #1e-5 #8e-5 #5e-5 #2e-5 #2e-4 #1e-4
    per_device_train_batch_size = 9 #5 #4 #6 #8 #10#6#16 #6 #2 #8#6 #8
    per_device_eval_batch_size =  9 #5 #4 #6 #8 #10 #6#16 #6 #2 #8#6 #8  
    # Lora config
    DROPOUT = 0.05 #0.1 #0.05
    weight_decay = 0.01 

    warmup_steps= 10 #5# 10
    gradient_accumulation_steps = 4 #8 # 4 #2  #10
    maxTrainStep =  500 #120 #160 #100 #50#100 #250 #150 
    valDatasetSize = 50 #20 #10 #50 #500 # set validiation data size

    #model
    model1 = "/kaggle/input/qwen-3/transformers/8b/1"
    model2 = "/kaggle/input/qwen-3-reranker/transformers/4b/1"

    #
    frac = 0.5 #0.3 #0.3 #0.3 #0.1 # Random resample data(reduce train data size )
    

    trainFile = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
    testFile  = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
    sampleFile = "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    deepspeedConfigFile = "/kaggle/input/deepspeed-config-for-kaggle-t4x2-gpu/deepspeed_config.json"
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


torch.cuda.is_bf16_supported()


def clearMemory():
    for _ in range(5):
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.3)


clearMemory()


# !accelerate config


trainDF = pd.read_csv(CFG.trainFile)
trainDF.head()


trainDF.info()


trainDF.describe()


testDF = pd.read_csv(CFG.testFile)
testDF


testDF.info()


testDF.isnull().sum()


# Nan and Null check
trainDF.isnull().sum()



sample = pd.read_csv(CFG.sampleFile)
sample


sample.columns


def printAllcolumnsValue(df, showAll=True):
    """
    Print DataFrame columns values
    """
    # loop column
    for col in df.columns:
        if showAll is True:
            print(f"{col} : {df[col].unique()}") # print unique value
        else: # only print caterogy column
            if df[col].dtype == "object":
                print(f"{col} : {df[col].unique()}") # print unique value

def printSerieUnquieValue(df):
    """
    print data serie value
    """
    print(f" {df.unique()}")


printAllcolumnsValue(trainDF, showAll=True)


featureCols = trainDF.columns.tolist()
featureCols


# clean data
trainDF["Misconception"]= trainDF["Misconception"].fillna("NA") # fill NaN to NA
trainDF["Misconception"].isnull().sum() # check Null 



trainDF['target'] = trainDF["Category"]+":"+trainDF["Misconception"]
trainDF


le = LabelEncoder()
trainDF["label"]= le.fit_transform(trainDF['target']) # encode caterogy into label


trainDF


# print all unique t
printAllcolumnsValue(trainDF, showAll=True)



targetClasses = le.classes_   
targetClasses


nClasses = len(targetClasses)
nClasses


print(f"Train shape: {trainDF.shape} with {nClasses} target classes")
trainDF.head()


idx = trainDF.apply(lambda row: row["Category"].split("_")[0] , axis=1) == "True" # get Category with True 
idx


correct = trainDF.loc[idx].copy()
correct


# group MC anwser with count 
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False) # sort value 
correct


# Drop duplicateds ID
correct = correct.drop_duplicates(['QuestionId']) #drop dprlicate id
correct = correct[['QuestionId','MC_Answer']]
correct


# set correct flag = 1
correct["is_correct"] = 1
correct



trainDF = trainDF.merge(correct, on=['QuestionId','MC_Answer'], how='left')
trainDF["is_correct"] = trainDF["is_correct"].fillna(0) # fill miss data to 0


trainDF["is_correct"] = trainDF["is_correct"].astype(int) # change is_correct data type to int


trainDF


trainDF["is_correct"].value_counts().plot(kind="bar", title="Correct vs Incorrect");


trainDF["label"].value_counts().plot(kind="bar", title="Distribution Label");


trainDF["label"].value_counts().describe()


trainDF["label"].value_counts()[:50]  # top 50


testDF = testDF.merge(correct, on=['QuestionId','MC_Answer'], how='left')
testDF["is_correct"] = testDF["is_correct"].fillna(0)  # fill Na to 0



testDF["is_correct"] = testDF["is_correct"].astype(int)
testDF


from IPython.display import display, Math, Latex


temp = trainDF.groupby(["QuestionId", "MC_Answer"]).size().reset_index(name="count")
temp


# add Ranking
temp["rank"] = temp.groupby("QuestionId")['count'].rank(method='dense', ascending=False).astype(int) - 1
temp


temp = temp.drop('count',axis=1)
temp = temp.sort_values(['QuestionId','rank'])
temp


Qid = temp["QuestionId"].unique() # get unique questionid
for q in Qid:
    question = trainDF.loc[trainDF.QuestionId==q].iloc[0]["QuestionText"] # get Question Text for specific QuestionId
    choices = temp.loc[temp["QuestionId"] ==q]["MC_Answer"].values #
    labels="ABCD" # define labels name
    choiceStr = " ".join([f"({labels[i]}) {choice}" for  i, choice in enumerate(choices)])
    print()
    display(Latex(f"Question Id: {q}: {question}"))
    display(Latex(f"MC Answer: {choiceStr}\n\r"))
    
    # print(f"question: {question}, choices: {choices}  \n\r")


len(Qid)


clearMemory()


sampleValue =  14802 //5
sampleValue


# median value = 50 
(trainDF["label"].value_counts() > sampleValue).sum() # number of class > 50 counts


# Assume df is your DataFrame and 'label' is your class column
desired_count = sampleValue #50
def resample_group_mix(group):
    '''
    Resample with hybird undersample and Oversample
    '''
    n = len(group)
    if n > desired_count:
        # Undersample if more than desired_count
        return group.sample(n=desired_count, random_state=42)
    elif n < desired_count:
        # Oversample with replacement if less than desired_count
        return group.sample(n=desired_count, replace=True, random_state=42)
    else:
        # Keep as is if exactly desired_count
        return group


# conditional oversample 
oversample_count = 1500 #1000
def resample_group_oversample(group):
    '''
    Resample with condition Oversample , keep major
    '''
    n = len(group)
    if n < oversample_count:
        # Oversample with replacement if less than desired_count
        return group.sample(n=oversample_count, replace=True, random_state=42)
    else:
        # Keep as is if exactly desired_count
        return group
    


#

# Group by class and sample min(desired_count, current_count) from each class
# trainDF_balanced = trainDF.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), desired_count), random_state=42))
trainDF_balanced = trainDF.groupby('label', group_keys=False).apply(resample_group_oversample)

print(trainDF_balanced['label'].value_counts())


trainDF_balanced['label'].value_counts()[:55]


trainDF_balanced['label'].value_counts().plot(kind="bar", title="Distribution Label");


trainDF_balanced = trainDF_balanced.reset_index(drop=True)


trainDF_balanced


trainDF_balanced.isnull().sum() # null checking


# !pip uninstall protobuf
# !pip install protobuf==3.20.3


bnbConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True ) # Activate netsed quantization for 4bit base 


# !pip install triton==3.4.0
# !pip install triton==3.0.0
# !pip install protobuf==6.30.2





clearMemory()


# !pip install triton==3.0.0
import trl
trl.__version__




import triton

triton.__version__


# !pip install bitsandbytes==0.46.1


import bitsandbytes
# bitsandbytes.__version__


# Mapping from label to ID (label2id)
label2id = {label: i for i, label in enumerate(le.classes_)}
print(f"Label to ID mapping: {label2id}")


# Mapping from ID to label (id2label)
id2label = {i: label for i, label in enumerate(le.classes_)}
print(f"ID to label mapping: {id2label}")


if device.type == "cuda":
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model2,  #CFG.model1, #CFG.model3,
        num_labels= nClasses,
        id2label =id2label,
        label2id = label2id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config= bnbConfig,
        trust_remote_code=False,
        problem_type="single_label_classification", 
    )

else: 
     model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model2,  #CFG.model1 #CFG.model3,
        num_labels= nClasses,
        id2label =id2label,
        label2id = label2id,
        device_map="auto",
        trust_remote_code=False,
        problem_type="single_label_classification", 
    )

tokenizer = AutoTokenizer.from_pretrained(CFG.model2)
tokenizer.pad_token = tokenizer.eos_token


model


if CFG.DeepEval:
    from deepeval.models.base_model import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval
    # from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric
    # from deepeval.benchmarks import MMLU, HellaSwag, HumanEval,TruthfulQA
    # from deepeval.benchmarks.tasks import HumanEvalTask, HellaSwagTask, MMLUTask, TruthfulQATask
    # from deepeval.benchmarks.modes import TruthfulQAMode 


# Random Resample Dataset
# trainDF = trainDF.sample(frac=1.0, replace=True)
# trainDF = trainDF.reset_index(drop=True)
# trainDF

trainDF_balanced= trainDF_balanced.sample(frac=1.0, replace=True)
trainDF_balanced = trainDF_balanced.reset_index(drop=True)
trainDF_balanced


PROMPT_FORMAT1 = """\
You are an expert in identifying math misconceptions from student explanations.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}

Identify the specific type of math misconception in the student's explanation.
Give a short, clear label for the misconception.
"""

PROMPT_FORMAT2 = """\
You are an expert educational diagnostician specializing in identifying students' math misconceptions from their written explanations.
Your task is to analyze the student's explanation and classify the specific type of misconception they demonstrate.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}

Based on the information above, identify and label the underlying math misconception in the student's explanation.
Provide a clear and concise classification of the misconception type.
"""

PROMPT_FORMAT3 = """\
You are an expert in analyzing students' answers to math problems and identifying specific types of misunderstandings.
Given the following information, classify the student's misunderstanding.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}

Please determine the type of misunderstanding the student has.
"""

PROMPT_FORMAT4 = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}
"""


def generateTrainContent(row):
    x= "True"
    if not row["is_correct"]:
        x= "False"
    return PROMPT_FORMAT2.format(
        QuestionText=row['QuestionText'],
        MC_Answer=row['MC_Answer'],
        Correct=x,
        StudentExplanation=row['StudentExplanation']
        
    )


# trainDF["LLM_Content"]= trainDF.apply(generateTrainContent, axis=1)
trainDF_balanced["LLM_Content"] = trainDF_balanced.apply(generateTrainContent, axis=1)


testDF["LLM_Content"]= testDF.apply(generateTrainContent, axis=1)


# print one of example
print(trainDF_balanced["LLM_Content"].values[0])


print(testDF["LLM_Content"].values[0])


trainDF_balanced["LLM_Content"].str.len().value_counts() # raw input character length


# trainDF["Length"] = trainDF["LLM_Content"].str.len()
trainDF_balanced["Length"] = [len(tokenizer.encode(t, truncation=False)) for t in trainDF_balanced["LLM_Content"]]

trainDF_balanced["Length"].describe()


trainDF_balanced["Length"].plot(kind='hist', bins=50, subplots=True,sharex=True,sharey=True, title='LLM Content Length Distribution');


MAX_LEN = 256 #512 #256 #512
L = (np.array(trainDF_balanced["Length"])>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")


len(trainDF_balanced) , round(len(trainDF_balanced) * 0.85) # check train data size


totalTrainSize= len(trainDF_balanced)
totalTrainSize


setMaxTrainData = totalTrainSize - CFG.valDatasetSize
setMaxTrainData






# tempTrainDF , tempValDF = train_test_split(trainDF_balanced , stratify= trainDF["is_correct"], test_size=CFG.test_size, random_state=42)
tempTrainDF , tempValDF = train_test_split(trainDF_balanced ,  test_size=CFG.test_size, random_state=42)


tempTrainDF


tempTrainDF["label"].value_counts().plot(kind="bar", title="Class Distribution");


tempValDF.describe()


# Convert to Hugging Face Dataset for training dataset
COLS = ['LLM_Content','label']

trainDataset = Dataset.from_pandas(tempTrainDF[COLS], split="train")
valDataset = Dataset.from_pandas(tempValDF[COLS], split="test")
testDataset = Dataset.from_pandas(testDF, split="test")
subDataset = Dataset.from_pandas(sample, split="test") # submit dataset

# trainDataset = Dataset.from_pandas(tempTrainDF, split="train")
# valDataset = Dataset.from_pandas(tempValDF, split="test")
# subDataset = Dataset.from_pandas(sample, split="test") # submit dataset


trainDataset


testDataset


subDataset


# for GPU only
!nvidia-smi  


del tempTrainDF 
del tempValDF


datasetDict = DatasetDict({
    "train" : trainDataset,
    "val"  : valDataset
    
})
datasetDict


datasetDict["train"]


def tokenizeFunc(batch):
    return tokenizer(batch["LLM_Content"],  max_length=CFG.maxLength, padding='max_length', truncation=True)


datasetDict = datasetDict.map(tokenizeFunc, batched=True)


datasetDict


datasetDict = datasetDict.rename_column("label", "labels")  # change name to labels 


datasetDict


testDict = DatasetDict({
    "test": testDataset
    
})
testDict


submitDict = DatasetDict({
    "test": subDataset
    # "test": subDataset
    
})
submitDict


# Evaluation Metrics 
def compute_metric(eval_pred):
    logits , labels = eval_pred
    # predictions = np.argmax(logits, axis=-1)  # for mult-class
    top3_preds = np.argsort(logits, axis=-1)[:, -3:]
    match = (top3_preds == labels[:, None])
    predictions= top3_preds[:, 2]
    acc = (predictions == labels).mean()
    f1score  =  f1_score(labels, predictions, average='weighted', zero_division=1)
    recallScore = recall_score(labels, predictions,  average='weighted', zero_division=1)
    precision = precision_score(labels, predictions,  average='weighted', zero_division=1)
    
    # for map@3
    top3_preds = np.argsort(logits, axis=-1)[:, -3:]
    match = (top3_preds == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    map3Score = map3 / len(labels)
    # map3List.append(map3Score)
    
    return {"accuracy": acc , "recall":  recallScore,  "precision":precision, 'f1-score': f1score ,
             "map_3":map3Score}
    
    
    
    


lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        task_type="SEQ_CLS",  # for Sequence Classification
        # task_type="CAUSAL_LM", # for generative task
        bias="none",
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout= CFG.DROPOUT,
    )


# Creat Lora Model
model = get_peft_model(model, lora_config)


model.print_trainable_parameters()


import trl
trl.__version__


 CFG.maxEpoch


N_SAMPLES = len(datasetDict["train"])  # Use train_df after the split
STEPS_PER_EPOCH = (N_SAMPLES // CFG.per_device_train_batch_size ) #* CFG.maxEpoch
VAL_STEPS_PER_EPOCH = len(datasetDict["val"]) // CFG.per_device_eval_batch_size
TOTAL_STEPS = int(STEPS_PER_EPOCH * CFG.maxEpoch)
NUM_WARMUP_STEPS = int(TOTAL_STEPS * 0.05)

print(f'BATCH_SIZE: {CFG.per_device_train_batch_size}, N_SAMPLES: {N_SAMPLES}, STEPS_PER_EPOCH: {STEPS_PER_EPOCH}')
print(f'NUM_WARMUP_STEPS: {NUM_WARMUP_STEPS}, TOTAL_STEPS: {TOTAL_STEPS}')
print(f"Val Step per epoch: {VAL_STEPS_PER_EPOCH}")



# import deepspeed
# deepspeed.__version__


import accelerate
accelerate.__version__


from trl import SFTTrainer, SFTConfig
torch_dtype = model.dtype
trainArg = SFTConfig(
    output_dir= "kaggle/working",
    # max_length= CFG.maxLength, # old version trl
    max_seq_length = CFG.maxLength, # new version trl
    per_device_train_batch_size =CFG.per_device_train_batch_size,
    per_device_eval_batch_size = CFG.per_device_eval_batch_size,
    gradient_accumulation_steps = CFG.gradient_accumulation_steps,
    eval_strategy= "steps",
    save_strategy= "steps",
    warmup_steps= NUM_WARMUP_STEPS, #CFG.warmup_steps,
    max_steps = TOTAL_STEPS,  #CFG.maxTrainStep, # set max train step size
    learning_rate=CFG.learning_rate,
    lr_scheduler_type="cosine", #"linear", #"cosine",
    save_total_limit=2,
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision ,  # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision, TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU  
    # fp16=True,
    logging_steps= CFG.evalSteps,
    eval_steps = CFG.evalSteps,
    optim="adamw_8bit",  #"paged_adamw_8bit",
    report_to = CFG.reportTo,
    do_eval=True,                # Perform evaluation at the end of training
    max_grad_norm=1.0,
    weight_decay=CFG.weight_decay,
    metric_for_best_model="map_3",
    greater_is_better=True,
    # memory optimial (reduce GPU memory but slowdown train speed )
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={'use_reentrant':False}, # for Mulitple -GPU
    load_best_model_at_end=True,
    # deepspeed=CFG.deepspeedConfigFile, # config error
    # label_names = targetClasses,

    
)


trainArg.bf16 


from transformers import DataCollatorWithPadding


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer= SFTTrainer(
    model=model,
    train_dataset=datasetDict["train"],
    eval_dataset = datasetDict["val"],
    args=trainArg,
    peft_config = lora_config,
    # formatting_func=formatFuct,
    data_collator = data_collator, 
    compute_metrics= compute_metric,
    
)


# Fix Batch size >1 , not define pad token id issues
eos_token_id=tokenizer.eos_token_id
model.config.use_cache=False
model.config.pad_token_id = tokenizer.pad_token_id


clearMemory() # clear memory before training


# %%time
trainer.train()


model.save_pretrained("student-math-llm-Fine-tune") # for peft fine tuning model without trainer
# trainer.save_model("student-math-llm-Fine-tune") # save model with full trainer 


# for generate downloadable zip file 
!zip -r student-math-llm-Fine-tune.zip /kaggle/working/student-math-llm-Fine-tune # for peft



trainHistoryDF = pd.DataFrame(trainer.state.log_history)
trainHistoryDF


trainHistoryDF.columns


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


trainloss = trainHistoryDF[~trainHistoryDF["eval_map_3"].isnull()]
plt.plot(trainloss["eval_map_3"], label="MAP@3 Score")
plt.title("Val MAP@3")
plt.legend()
plt.show()


def testValDataset(ds, maxNumData=10):
    for i, data in enumerate(ds):
        if (i >= maxNumData):
            break
        print(f"Data {i}")
        newPrompt = data["LLM_Content"]
        with torch.no_grad():
             inputIds = tokenizer(newPrompt, return_tensors="pt").to(device)
             logits = model(**inputIds).logits
             print(f"Logits Dimension: {logits.shape}")
             probabilities = nn.functional.softmax(logits, dim=-1) # get probilitity
             print(f"Probailities : {probabilities}")
             classID =logits.argmax().item()
             print(f"Class ID type {type(classID)}")
             classTxt = le.inverse_transform([classID])[0]
             actualClassID = data['labels']
             # print(f"Actual ClassID : {actualClassID}")
             actualClass = le.inverse_transform([actualClassID])[0]
             # Top3 prediction
             # acc = (probabilities == actualClassID).mean()
             # f1score  =  f1_score(actualClassID, probabilities, average='weighted', zero_division=1)
             # recallScore = recall_score(actualClassID, probabilities,  average='weighted', zero_division=1)
             # precision = precision_score(actualClassID, probabilities,  average='weighted', zero_division=1)
            
             # # for map@3
             top3_preds = np.argsort(logits.to(torch.float32).cpu(), axis=-1)[:, -3:]
             top3 = top3_preds.flatten() # convert to 1D 
             print(f"Top3 ID : {top3}")
             # classTxtArray = le.inverse_transform(top3) # convert class ID to Label 
             match = (top3_preds == actualClassID)
                    
             # Compute MAP@3 manually
             map3 = 0
             for i in range(len([actualClassID])):
                if match[i, 0]:
                    map3 += 1.0
                elif match[i, 1]:
                    map3 += 1.0 / 2
                elif match[i, 2]:
                    map3 += 1.0 / 3
             map3Score = map3 / len([actualClassID])
        

        print(f"""Query: {newPrompt}
              Predict Class ID: {classID}
              Predict Text: {classTxt}
              Actual Class ID: {actualClassID}
              Actual Class Text: {actualClass}
             MAP@3 Score : {map3Score}""")
        print("-"*30)
        



testValDataset(datasetDict["val"], 20 )





































































































































