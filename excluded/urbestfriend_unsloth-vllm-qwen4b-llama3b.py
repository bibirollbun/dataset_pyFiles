# unsloth/qwen3_4b_instruct2507_singleT4
# æµ�å¼�rankï¼Œæ”¾å¼ƒå¹¶è¡Œå�Œgpu loraå¼‚æ�„ï¼Œ é¢„è®¡OOMèŒƒå›´>=0.3ï¼Œå¹¶è¡Œæ��å®¹æ˜“OOMï¼Œæµ�å¼�æ—¶é—´å……åˆ†ï¼Œæœ¬åœ°å®‰å…¨


# å�•qwen-4B rank128 lr1-e4 0.913 
# æ�¨æµ‹ è¿›ä¸€æ­¥å¢�å¤§ rank


# r256 qwen3 4b T4*1 unsloth çˆ†æ˜¾å­˜
# æµ‹è¯•r1923


%%writefile Monkey_llm.py

Monkey_llm=False # if False use qwen


%%writefile download_unsloth_llm.py

# @title dowload unsloth llm

from unsloth import FastLanguageModel
import torch

max_seq_length = 256 # Choose any! We auto support RoPE Scaling internally! #æ¨¡å�‹èƒ½åŠ›ä¸Šé™�
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+


# è‡ªåŠ¨æŒ‚è½½gpuä¸Š
def load_model(model_name:str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = False,
    )
    return model, tokenizer


%%writefile constants.py


import importlib
import Monkey_llm
importlib.reload(Monkey_llm)
from Monkey_llm import Monkey_llm



#merged_path
merged_qwen_path = "/kaggle/working/merged_qwen_path"
merged_llama_path= "/kaggle/working/merged_llama_path"
# è®°å¾—è·‘å®Œåˆ é™¤å�ˆå¹¶å��çš„æ¨¡å�‹

#BASE_MODEL_PATH 
qwen_path='/kaggle/input/unslothqwen3-4b-instruct/other/default/1/Qwen3-4B-Instruct-2507'# offered by unsloth
llama_path='/kaggle/input/unslothllama-3-2-3b-instruct-end/other/default/1/Llama-3.2-3B-Instruct'

DATA_PATH = '/kaggle/input/jigsaw-agile-community-rules/'#"/kaggle/input/jigsaw-agile-community-rules/"
POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"# é”šç‚¹æ�©ç �token
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."

CLEAN_TEXT = True


if Monkey_llm:
    merged_qwen_path = merged_llama_path
    qwen_path= llama_path





%%writefile GET_SFT_DATASET.py


# @title æ�„é€  SFT propmt_completion
# è¿™è¾¹å�¯èƒ½éœ€è¦�æ¸…æ´—
# openai_style

import importlib
import constants
importlib.reload(constants)
from constants import POSITIVE_ANSWER, NEGATIVE_ANSWER, COMPLETE_PHRASE, BASE_PROMPT, DATA_PATH, CLEAN_TEXT

import pandas as pd
from datasets import Dataset
from cleantext import clean


# å¼‚æ�„ prompt
def build_prompt(row):
    return f"""
{BASE_PROMPT}

r/{row["subreddit"]} rule: {row["rule"]}

Comment: {row["body"]}
---
{COMPLETE_PHRASE}"""



def cleaner(text):
    return clean(
        text,
        fix_unicode=False,
        to_ascii=False,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        # lang="en",
    )




def get_dataframe_to_train(data_path):
    train_dataset = pd.read_csv(f"{data_path}/train.csv")
    test_dataset = pd.read_csv(f"{data_path}/test.csv")

    flatten = []
    # flatten.append(train_dataset[["body", "rule", "subreddit", "rule_violation"]])
    # #æ��äº¤éœ€ä¿®æ”¹æ•°æ�®è�·å�–
    # for violation_type in ["positive", "negative"]:
    #     for i in range(1, 3):
    #         sub_dataset = train_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
    #         sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
    #         sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
    #         flatten.append(sub_dataset)

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            sub_dataset = test_dataset[[f"{violation_type}_example_{i}", "rule", "subreddit"]].copy()
            sub_dataset = sub_dataset.rename(columns={f"{violation_type}_example_{i}": "body"})
            sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
            flatten.append(sub_dataset)
            
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(ignore_index=True)
    dataframe = dataframe.sample(frac=1, random_state=3407).reset_index(drop=True)# T
    return dataframe

def build_dataset(dataframe):
    '''
    if clean, clean before this
    è¿™é‡Œè€ƒè™‘å¼‚æ�„æ¨¡æ�¿æ�¨ç�†# ä¸�ç”¨è€ƒè™‘äº†
    '''

    # å�ªæ¸…ç�†body ä¸�æ¸…ç�†æ¨¡æ�¿
    if CLEAN_TEXT:
        # tqdm.pandas(desc="cleaner")
        dataframe["body"] = dataframe["body"].apply(cleaner)


    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)

    columns = ["prompt"]
    if "rule_violation" in dataframe:
        dataframe["completion"] = dataframe["rule_violation"].map(
            {
                1: POSITIVE_ANSWER,
                0: NEGATIVE_ANSWER,
            }
        )
        columns.append("completion")




    dataframe = dataframe[columns]
    dataset = Dataset.from_pandas(dataframe)
    return dataset





%%writefile set_args_unsloth_llm.py
# ä¹Ÿè®¸ä¸¤ä¸ªå¼‚æ�„ä¸�è¯¥è®¾ç½®ä¸€æ ·
# qwenæ›´é«˜
# @title set_lora_args_unsloth_llm
import importlib
import download_unsloth_llm
importlib.reload(download_unsloth_llm)
from download_unsloth_llm import load_model
from unsloth import FastLanguageModel

def LoRA_Adapter_Unsloth(model_name:str):
    model,tokenizer = load_model(model_name) # model_name='unsloth/Qwen3-4B-Instruct-2507-bnb-4bit'
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,#192 for llama # high for qwen? # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 256, # 
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 9527,#955 #996 #666 #99999 #12127 #24247 #3407
        #task_type default
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ


    )
    return model,tokenizer


%%writefile dataset_out_router.py

# @title dataset_out_router
# æœ¬åœ°æµ‹è¯•çš„æ•°æ�®é›†è·¯ç”±
import importlib
import GET_SFT_DATASET
importlib.reload(GET_SFT_DATASET)
from GET_SFT_DATASET import get_dataframe_to_train, build_dataset


import constants
importlib.reload(constants)
from constants import DATA_PATH

from sklearn.model_selection import train_test_split
import pandas as pd


dataframe=get_dataframe_to_train(DATA_PATH)# dataset

# æµ‹è¯•å��ä¹‰prompt
# dataframe['rule_violation']=dataframe['rule_violation'].apply(lambda x:1 if x==0 else 0)
# dataframe=dataframe[dataframe['subreddit']== subredit_name ].reset_index(drop=True)


# train_df for train
# val_df for val
# test_df dropout
train_df, test_df = train_test_split(
    dataframe,
    test_size=0.8, # val_a # should be modified with below val_b to define fact train_df size  
    #random_state=args.random_state,
    shuffle=True
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.5, # val_b # 
    #random_state=args.random_state,  fact train_df size  account for (1-val_a)* (1-val_b)
    shuffle=True
)


train_df.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)



%%writefile unsloth_train.py
# @title train unsloth
# unsloth & vllm need def to recycle

from unsloth import is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available

from download_unsloth_llm import max_seq_length
from set_args_unsloth_llm import LoRA_Adapter_Unsloth

import importlib
import GET_SFT_DATASET
importlib.reload(GET_SFT_DATASET)
from GET_SFT_DATASET import  build_dataset


import constants
importlib.reload(constants)
from constants import merged_qwen_path, qwen_path, DATA_PATH

import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import os

# âœ… è¯»å�– DataFrame, åº”è¯¥åŒ…å�«ä¸¤åˆ—: prompt / completion
#  pipeline

train_df=pd.read_csv("train_df.csv")
dataset=build_dataset(train_df)

# load model, tokenizer
model, tokenizer = LoRA_Adapter_Unsloth(qwen_path)
# tokenize and mask
def preprocess_function(example):
    # make sure completion end with EOS
    full_text = example["prompt"] + example["completion"] + tokenizer.eos_token
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
        padding=False,
        return_tensors=None,
    )
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    prompt_end_char = len(example["prompt"])
    labels = [
        -100 if start < prompt_end_char else token_id
        for (start, _), token_id in zip(offsets, input_ids)
    ]
    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = dataset.map(
    preprocess_function,
    remove_columns=["prompt", "completion"],  # remove used columns
    desc="Tokenizing and masking",
)




# å¤�åˆ¶äº�å…¬å¼€lora 0.5B instructè„šæœ¬å�‚æ•°
# å�‚è€ƒunsloth qwen ç¤ºä¾‹è¿›ä¸€æ­¥ä¿®æ­£
training_args = SFTConfig(
    num_train_epochs=1,
    per_device_train_batch_size=16,#packing #16
    gradient_accumulation_steps=2,#packing
    optim="paged_adamw_8bit",
    learning_rate=1e-4, #5e-5 # rank64 æ�¨è�� 3e-4
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    #dataset_text_field='prompt',
    #completion_only_loss=True,# True

    bf16=is_torch_bf16_gpu_available(),
    fp16=not is_torch_bf16_gpu_available(),

    dataloader_pin_memory=True, #å†…å­˜ #Falseè¾¹ç¼˜
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    save_strategy="no",
    report_to="none",


    packing= False,#False,# True ä¼šç ´å��å·²ç¼–ç �åº�åˆ—
    remove_unused_columns=False,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    args=training_args,
    # dataset_num_proc=1, #è¿›ç¨‹æ•°1 #RAM 3.4G gpu 5G # >1 resort by length and influence patch
)

def main():
    trainer.train()

    # 1. ç¡®ä¿�æ¨¡å�‹å¤„äº� eval æ¨¡å¼�ï¼ˆé�¿å…� dropout ç­‰å½±å“�ï¼‰
    model.eval()
    merged_model = model.merge_and_unload()  # è¿”å›�ä¸€ä¸ªæ™®é€š transformers æ¨¡å�‹
    merged_model = merged_model.to(torch.float16)

    # 4. ä¿�å­˜å�ˆå¹¶å��çš„æ¨¡å�‹ï¼ˆçº¯ fp16ï¼Œæ—  int8ï¼‰
    merged_model.save_pretrained(
        merged_qwen_path,
        safe_serialization=True,
        max_shard_size="5GB"  # å�¯é€‰ï¼šé�¿å…�å�•æ–‡ä»¶è¿‡å¤§
    )
    tokenizer.save_pretrained(merged_qwen_path)

    print(f"âœ… Merged model saved to {merged_qwen_path} in fp16")




if __name__ == "__main__":
    main()



%%writefile inference.py

import os
os.environ["VLLM_USE_V1"] = "0"
import numpy as np
from math import exp

import vllm
import torch
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from datasets import Dataset
# from utils import build_dataset

import importlib
import constants
importlib.reload(constants)
from constants import  POSITIVE_ANSWER, NEGATIVE_ANSWER, DATA_PATH, merged_qwen_path,Monkey_llm

import GET_SFT_DATASET
importlib.reload(GET_SFT_DATASET)
from GET_SFT_DATASET import get_dataframe_to_train, build_dataset





def main():

    # âœ… è¯»å�– DataFrame, åº”è¯¥åŒ…å�«ä¸¤åˆ—: prompt / completion
    # df = get_dataframe_to_train(DATA_PATH)[1500:]# T
    # testset = build_dataset(df)
    # texts=testset[:]['prompt']
    df = pd.read_csv(f"{DATA_PATH}/test.csv")

    
    testset=build_dataset(df)
    texts=testset[:]['prompt']


    llm = vllm.LLM(

        merged_qwen_path,
        # quantization="gptq", #t
        tensor_parallel_size=1,#torch.cuda.device_count(), see issue https://github.com/vllm-project/vllm/issues/2699?ysclid=me2q9w8mby837333215
        gpu_memory_utilization=0.85,#
        trust_remote_code=True,
        dtype="auto",#

        enforce_eager=True,#True
        max_model_len=1024,#4096
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=False, #é‡‡ç”¨å�ˆå¹¶å��çš„æ¨¡å�‹
    )


    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE_ANSWER, NEGATIVE_ANSWER])

    print('llm load succeed')


    outputs = llm.generate(
        texts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
    )

    print('GENERATE SUCCEED')

    log_probs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]

    print('LOG PROB SUCCEED')
    predictions = pd.DataFrame(log_probs)[[POSITIVE_ANSWER, NEGATIVE_ANSWER]]
    
    predictions["row_id"] = df["row_id"].values

    submission=predictions[["row_id", POSITIVE_ANSWER]].rename(columns={POSITIVE_ANSWER: "rule_violation"})
    
    # submission['rule_violation']=np.exp(submission['rule_violation'])
    
    if Monkey_llm:
        submission.to_csv("llama.csv", index=False)
        print('build llama.csv !')
    else:
        submission.to_csv("qwen.csv", index=False)
        print('build qwen.csv !')

    print('ALL succeed')
    #return predictions

if __name__ == "__main__":
    main()


%%writefile val_inference.py

import os
os.environ["VLLM_USE_V1"] = "0"
import numpy as np
from math import exp

import vllm
import torch
import pandas as pd
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from datasets import Dataset
# from utils import build_dataset

import importlib
import constants
importlib.reload(constants)
from constants import  POSITIVE_ANSWER, NEGATIVE_ANSWER, DATA_PATH, merged_qwen_path,Monkey_llm

import GET_SFT_DATASET
importlib.reload(GET_SFT_DATASET)
from GET_SFT_DATASET import get_dataframe_to_train, build_dataset





def main():

    # âœ… è¯»å�– DataFrame, åº”è¯¥åŒ…å�«ä¸¤åˆ—: prompt / completion
    # df = get_dataframe_to_train(DATA_PATH)[1500:]# T
    # testset = build_dataset(df)
    # texts=testset[:]['prompt']
    df = pd.read_csv("val_df.csv")

    
    testset=build_dataset(df)
    texts=testset[:]['prompt']


    llm = vllm.LLM(

        merged_qwen_path,
        # quantization="gptq", #t
        tensor_parallel_size=1,#torch.cuda.device_count(), see issue https://github.com/vllm-project/vllm/issues/2699?ysclid=me2q9w8mby837333215
        gpu_memory_utilization=0.85,#
        trust_remote_code=True,
        dtype="auto",#

        enforce_eager=True,#True
        max_model_len=1024,#4096
        disable_log_stats=True,
        enable_prefix_caching=True,
        enable_lora=False, #é‡‡ç”¨å�ˆå¹¶å��çš„æ¨¡å�‹
    )


    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE_ANSWER, NEGATIVE_ANSWER])

    print('llm load succeed')


    outputs = llm.generate(
        texts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2,
        ),
        use_tqdm=True,
    )

    print('GENERATE SUCCEED')

    log_probs = [
        {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()}
        for out in outputs
    ]

    print('LOG PROB SUCCEED')
    predictions = pd.DataFrame(log_probs)[[POSITIVE_ANSWER, NEGATIVE_ANSWER]]
    

    if Monkey_llm:
        predictions.to_csv("val_llama.csv", index=False)
        print('build val_llama.csv !')
    else:
        predictions.to_csv("val_qwen.csv", index=False)
        print('build val_qwen.csv !')

    print('ALL succeed')
    #return predictions

if __name__ == "__main__":
    main()



!python dataset_out_router.py


!python unsloth_train.py


!python inference.py
!python val_inference.py


# åˆ é™¤ merged_qwen æ¨¡å�‹
import importlib
import constants
importlib.reload(constants)
from constants import  merged_qwen_path

import shutil

shutil.rmtree(merged_qwen_path)


%%writefile Monkey_llm.py
# æ³¨æ„�é¡ºåº�
Monkey_llm=True # if True use llama


%%writefile set_args_unsloth_llm.py
# ä¹Ÿè®¸ä¸¤ä¸ªå¼‚æ�„ä¸�è¯¥è®¾ç½®ä¸€æ ·
# qwenæ›´é«˜
# @title set_lora_args_unsloth_llm
import importlib
import download_unsloth_llm
importlib.reload(download_unsloth_llm)
from download_unsloth_llm import load_model
from unsloth import FastLanguageModel

def LoRA_Adapter_Unsloth(model_name:str):
    model,tokenizer = load_model(model_name) # model_name='unsloth/Qwen3-4B-Instruct-2507-bnb-4bit'
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 96,#192 for llama # high for qwen? # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 192, # 
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 12127,#955 #996 #666 #99999 #12127 #24247 #3407
        #task_type default
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ


    )
    return model,tokenizer


!python unsloth_train.py


!python inference.py
!python val_inference.py


# åˆ é™¤ merged_llama æ¨¡å�‹
import importlib
import constants
importlib.reload(constants)
from constants import  merged_qwen_path


import shutil

shutil.rmtree(merged_qwen_path)


%%writefile grid_rank.py

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from math import exp
import numpy as np
import pandas as pd

val_df=pd.read_csv('val_df.csv')

def get_auc(predictions):
    True_predictions=predictions.copy()
    # True_predictions['Yes']=np.exp(True_predictions['Yes'])
    
    #print('AUC', roc_auc_score(test_df['rule_violation'], True_predictions['Yes']))
    return roc_auc_score(val_df['rule_violation'], True_predictions['Yes'])




qwen = pd.read_csv('qwen.csv', dtype={'row_id': 'int64'})
llama = pd.read_csv('llama.csv', dtype={'row_id': 'int64'})

val_qwen=pd.read_csv('val_qwen.csv')
val_llama=pd.read_csv('val_llama.csv')


rqwen = val_qwen['Yes'].rank(method='average') / (len(val_qwen)+1)
rllama = val_llama['Yes'].rank(method='average') / (len(val_llama)+1)
RESULT=[]

i=0.1
while i<1:
    blend = i*rqwen + (1-i)*rllama
    temp_AUC=get_auc(blend.to_frame())
    RESULT.append(  { 'i':i, 'j':1-i, 'AUC':temp_AUC} )
    i+=0.1
df = pd.DataFrame(RESULT)
df = df.sort_values('AUC', ascending=False).reset_index(drop=True)

best_i=df['i'][0]
best_j=df['j'][0]
best_auc=df['AUC'][0]

# Blend test predictions (only on 'Yes' column!)
blend_test = best_i * qwen['rule_violation'] + best_j * llama['rule_violation'] #  tiny grid

# Construct submission with original row_id
submission = pd.DataFrame({
    'row_id': qwen['row_id'],
    'rule_violation': blend_test
})


submission.to_csv('submission.csv', index=False)

print(f"qwen best_i={best_i}, llama best_j={best_j}, best_auc={best_auc}")


!python grid_rank.py


!head submission.csv




