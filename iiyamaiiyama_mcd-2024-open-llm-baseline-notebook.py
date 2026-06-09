!pip install -q -U torch --index-url https://download.pytorch.org/whl/cu117
!pip install -q -U -i https://pypi.org/simple/ bitsandbytes
!pip install -q -U transformers
!pip install -q -U accelerate


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_name = "/kaggle/input/qwen2/transformers/qwen2-7b-instruct/1"


compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 1024
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
EOS_TOKEN = tokenizer.eos_token


input_text = "Hello! Who are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200, eos_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import pandas as pd
from tqdm import tqdm

data_dir = "/kaggle/input/mcd-data-science-competition-2024-open/"

article_path = data_dir + "article.csv"
sample_submission_path = data_dir + "sample_submission.csv"
test_path = data_dir + "test.csv"
train_path = data_dir + "train.csv"

article_df = pd.read_csv(article_path)
sample_submission_df = pd.read_csv(sample_submission_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)

article_df.shape,sample_submission_df.shape,test_df.shape,train_df.shape


def gen_prompt(row):
    ret = f"""
Please answer the following multiple-choice question. Output only the index of the correct answer.

## Question
{row.quiz_text}
0: {row.choice_0}
1: {row.choice_1}
2: {row.choice_2}
3: {row.choice_3}

## Answer
"""
    return ret


def extract_answer(text):
    # "<end_of_turn>"以降で最初に出現した数字を返す
    # 数字が出現しなかったら-1を返す

    try:
        index = text.index("## Answer")
    except:
        return 0
    ans = text[index+len("## Answer"):]
    
    for c in ans:
        if c.isdigit():
            return int(c)
    return -1


df = train_df

test_ret_lis = []
for i,row in tqdm(df.iterrows(),total=len(df)):
    
    p = gen_prompt(row)
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=len(inputs[0])+30, eos_token_id=tokenizer.eos_token_id ,pad_token_id=tokenizer.eos_token_id)

    ret = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ans = extract_answer(ret)

    test_ret_lis.append(ans)


train_df["pred"] = test_ret_lis

train_df.head(5)


len(train_df.query("answer == pred"))


len(train_df.query("answer == pred"))/len(train_df)





df = test_df

test_ret_lis = []
for i,row in tqdm(df.iterrows(),total=len(df)):
    
    p = gen_prompt(row)
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=len(inputs[0])+30, eos_token_id=tokenizer.eos_token_id ,pad_token_id=tokenizer.eos_token_id)

    ret = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ans = extract_answer(ret)

    test_ret_lis.append(ans)


test_df["answer"] = test_ret_lis
test_df.head()


test_df[["quiz_id","answer"]].to_csv("submission.csv",index=None)

