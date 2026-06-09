%%time
!pip uninstall -y torch
!pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
!pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q -U --upgrade /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
!pip install -q --no-deps --no-index /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.1.0-py3-none-any.whl



import os, math, numpy as np
import os
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re, gc
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
pd.set_option('display.max_rows', 300)



IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
df_train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv").fillna(-1).sample(100, random_state=42).reset_index(drop=True)
df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")



import pandas as pd
from sentence_transformers import SentenceTransformer, util

if not IS_SUBMISSION:
    df_ret = df_train.copy()
else:
    df_ret = df_test.copy()

df_misconception_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

model = SentenceTransformer('/kaggle/input/eedi-finetuned-bge-public/Eedi-finetuned-bge')
df_ret.head()



# Converts text to lowercase, removes URLs, leftover LaTeX escapes, repeated punctuation
def preprocess_text(x):
    x = x.lower()
    x = re.sub("@\w+", '',x)
    x = re.sub("http\w+", '',x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\times+", "\\\\times", x)
    x = x.strip()
    return x



firstPROMPT  = """
You are a Mathematics master. Your task is to succinctly summarize the core math knowledge from the information. 
Use your own words to convey the essence without directly referencing the problem's text, especially number in the text.

Information：
Here is a question about {ConstructName}({SubjectName}).
Question: {Question}

Example:
Information：
Here is a question about Simplify an algebraic fraction by factorising the numerator(BIDMAS).
Question: [3 x 2+4-5] Where do the brackets need to go to make the answer equal (13)?
Your answer: Understanding and applying the order of operations to achieve a specific result.
"""

def apply_keywordTemplate(row, tokenizer):
    messages = [
        {
            "role": "user", 
            "content": preprocess_text(
                firstPROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                )
            )
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

df_keyword = {}
for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
    df_keyword[f"{row.QuestionId}"] = apply_keywordTemplate(row, tokenizer)

df_keyword = pd.DataFrame([df_keyword]).T.reset_index()
df_keyword.columns = ["QuestionId", "text"]
df_keyword.to_parquet("forKeyword.parquet", index=False)
print(df_keyword.loc[0, 'text'])



%%writefile run_vllmKeyword.py
import re
import vllm
import pandas as pd

df = pd.read_parquet("forKeyword.parquet")
model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

llm = vllm.LLM(
    model_path,
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True
)
tokenizer = llm.get_tokenizer()

responses = llm.generate(
    df["text"].values,
    vllm.SamplingParams(
        n=1,
        top_p=0.8,
        temperature=0.1,
        seed=777,
        skip_special_tokens=False,
        max_tokens=1024
    ),
    use_tqdm=True
)

responses = [x.outputs[0].text for x in responses]
df["fullLLMText"] = responses

def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

df["llmKeyword"] = responses
df.to_parquet("forKeyword.parquet", index=False)



!python run_vllmKeyword.py


llm_output = pd.read_parquet("forKeyword.parquet")

for idx, row in llm_output[0:5].iterrows():
    print(row.llmKeyword)
    print("==="*6)

for idx, row in llm_output[0:5].iterrows():
    print(row.text)
    print("==="*6)

def clean_keyword(row):
    flag = 0
    if "your answer:" in row['llmKeyword']:
        flag = 1
    text = row['llmKeyword'].strip()
    if flag:
        text = text.replace("your answer:", "").strip()
    return text

llm_output['llmKeywordCleaned'] = llm_output.apply(clean_keyword, axis=1)

for idx, row in llm_output[0:5].iterrows():
    print(row.llmKeywordCleaned)
    print("==="*6)



df_ret['input_features'] = df_ret["ConstructName"] + ". " + df_ret["SubjectName"] + ". " + llm_output['llmKeywordCleaned']
df_ret['input_features'] = df_ret['input_features'].apply(lambda x: preprocess_text(x))

embedding_query = model.encode(df_ret['input_features'], convert_to_tensor=True)
misconceptions = df_misconception_mapping.MisconceptionName.values
embedding_Misconception = model.encode(misconceptions, convert_to_tensor=True)

Ret_topNids = util.semantic_search(embedding_query, embedding_Misconception, top_k=100)



retrivals = []
dicts = {}
for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
    top_ids = Ret_topNids[idx]
    retrival = ''
    dicts[str(row['QuestionId'])] = {}
    for i, ids in enumerate(top_ids):
        retrival += f'{i+1}. ' + misconceptions[ids['corpus_id']] + '\n'
        dicts[str(row['QuestionId'])][str(i+1)] = misconceptions[ids['corpus_id']]
    retrivals.append(retrival)

df_ret['Retrival'] = retrivals



df_ret.head()


def preprocess_text(x):
    x = re.sub("http\w+", '',x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = re.sub(r"\times+", "\\\\times", x)
    x = x.strip()
    return x

PROMPT  = """Here is a question about {ConstructName}({SubjectName}).
Question: {Question}
Correct Answer: {CorrectAnswer}
Incorrect Answer: {IncorrectAnswer}

You are a Mathematics teacher. Your task is to reason and identify the misconception behind the Incorrect Answer with the Question.
Answer concisely what misconception it is to lead to getting the incorrect answer.
No need to give the reasoning process and do not use "The misconception is" to start your answers.
There are some relative and possible misconceptions below to help you make the decision:

{Retrival}
"""

def apply_template(row, tokenizer, targetCol):
    messages = [
        {
            "role": "user",
            "content": preprocess_text(
                PROMPT.format(
                    ConstructName=row["ConstructName"],
                    SubjectName=row["SubjectName"],
                    Question=row["QuestionText"],
                    IncorrectAnswer=row[f"Answer{targetCol}Text"],
                    CorrectAnswer=row[f"Answer{row.CorrectAnswer}Text"],
                    Retrival=row["Retrival"]
                )
            )
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text

df = {}
if not IS_SUBMISSION:
    df_label = {}
    for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
        for option in ["A", "B", "C", "D"]:
            if (row.CorrectAnswer!=option) & (row[f"Misconception{option}Id"]!=-1):
                df[f"{row.QuestionId}_{option}"] = apply_template(row, tokenizer, option)
                df_label[f"{row.QuestionId}_{option}"] = [row[f"Misconception{option}Id"]]
    df_label = pd.DataFrame([df_label]).T.reset_index()
    df_label.columns = ["QuestionId_Answer", "MisconceptionId"]
    df_label.to_parquet("label.parquet", index=False)
else:
    for idx, row in tqdm(df_ret.iterrows(), total=len(df_ret)):
        for option in ["A", "B", "C", "D"]:
            if row.CorrectAnswer!=option:
                df[f"{row.QuestionId}_{option}"] = apply_template(row, tokenizer, option)

df = pd.DataFrame([df]).T.reset_index()
df.columns = ["QuestionId_Answer", "text"]
df.to_parquet("submission.parquet", index=False)
print(df.loc[0, 'text'])



%%writefile run_vllm.py
import re
import vllm
import pandas as pd

df = pd.read_parquet("submission.parquet")
model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"

llm = vllm.LLM(
    model_path,
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    dtype="half",
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True
)
tokenizer = llm.get_tokenizer()

responses = llm.generate(
    df["text"].values,
    vllm.SamplingParams(
        n=1,
        top_p=0.8,
        temperature=0.1,
        seed=777,
        skip_special_tokens=False,
        max_tokens=1024
    ),
    use_tqdm=True
)

responses = [x.outputs[0].text for x in responses]
df["fullLLMText"] = responses

def extract_response(text):
    return ",".join(re.findall(r"<response>(.*?)</response>", text)).strip()

df["llmMisconception"] = responses
df.to_parquet("submission.parquet", index=False)


!python run_vllm.py


llm_output = pd.read_parquet("submission.parquet")

# Prints first 5 LLM outputs.
for idx, row in llm_output[0:5].iterrows():
    print(row.llmMisconception)
    print("==="*6)

text = llm_output.loc[0, 'text']
PREFIX = "<|im_start|>user"
text = text.split(PREFIX)[1].split("You are a Mathematics teacher.")[0].strip('\n').split('Here is a question about')[-1].strip()
print(text)



import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_parquet("submission.parquet")
df_misconception_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

model = SentenceTransformer('/kaggle/input/eedi-finetuned-bge-public/Eedi-finetuned-bge')

def number2sentence(row):
    text = row['llmMisconception'].strip()
    potential = re.search(r'^\w+\.{0,1}', text).group()
    if '.' in potential:
        sentence = text.replace(potential, '').strip()
    else:
        sentence = text
    return sentence

df['llmMisconception_clean'] = df.apply(number2sentence, axis=1)



PREFIX = "<|im_start|>user"
df['input_features'] = df["text"].apply(lambda x: x.split(PREFIX)[1].split("You are a Mathematics teacher.")[0].strip('\n').split('Here is a question about')[-1].strip())
df['input_features'] = df['input_features'].apply(lambda x: preprocess_text(x))
df['input_features']
df["llmMisconception_clean"]



embedding_query = model.encode(df['input_features'], convert_to_tensor=True)
embedding_Misconception = model.encode(df_misconception_mapping.MisconceptionName.values, convert_to_tensor=True)
top25ids1 = util.semantic_search(embedding_query, embedding_Misconception, top_k=25)
print(len(top25ids1))

embedding_query = model.encode(df["llmMisconception_clean"], convert_to_tensor=True)
embedding_Misconception = model.encode(df_misconception_mapping.MisconceptionName.values, convert_to_tensor=True)
top25ids2 = util.semantic_search(embedding_query, embedding_Misconception, top_k=25)



# Merges the two sets of top 25 results into one combined list for each row.
top25ids = []
for i in range(len(top25ids1)):
    top25id = top25ids1[i] + top25ids2[i]
    top25ids.append(top25id)
len(top25ids)



# Sorts them by descending score.
Cuts off at 25 or until the score drops below 0.6
new_top25ids = []
for top25id in top25ids:
    top25idss = sorted(top25id, key=lambda x: x['score'], reverse=True)
    new_top25id = []
    for item in top25idss :
        if len(new_top25id) >= 25:
            break
        if item['score'] >= 0.6:
            new_top25id.append(item)
        else:
            break
    new_top25ids.append(new_top25id)



df["MisconceptionId"] = [" ".join([str(x["corpus_id"]) for x in top25id]) for top25id in new_top25ids]
df[["QuestionId_Answer", "MisconceptionId"]].to_csv("submission.csv", index=False)
df.head()


