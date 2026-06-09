%%time
!pip uninstall -y torch
!pip install -q --no-index --find-links=/kaggle/input/making-wheels-of-necessary-packages-for-vllm vllm
!pip install -q -U /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q -U /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl
!pip install -q --no-deps --no-index /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.1.0-py3-none-any.whl
!pip install --no-deps --no-index /kaggle/input/logits-processor-zoo/logits_processor_zoo-0.1.0-py3-none-any.whl


!pip install transformers peft accelerate \
    -q -U --no-index --find-links /kaggle/input/lmsys-wheel-files


%%capture
!pip install --no-index /kaggle/input/bitsandbytes0-42-0/bitsandbytes-0.42.0-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
!pip install --no-index  /kaggle/input/bitsandbytes0-42-0/optimum-1.21.2-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
!pip install --no-index  /kaggle/input/bitsandbytes0-42-0/auto_gptq-0.7.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --find-links=/kaggle/input/bitsandbytes0-42-0


# !pip install flash-attn --no-build-isolation


!pip install transformers==4.51.0


import pandas as pd
pd.read_csv('/kaggle/input/tianchi-external-data/module_extract.csv').head()





%%writefile get_sentence_match_pair.py
import json
import pandas as pd
from typing import List
import tqdm
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


import pandas as pd
external_info = pd.read_csv('/kaggle/input/tianchi-external-data/module_extract.csv')
external_info['matched_sentences'] = external_info['matched_sentences'].apply(lambda x: eval(x))
external_info


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/qwen-3-embedding/transformers/0.6b/1', padding_side='left')

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = AutoModel.from_pretrained('/kaggle/input/qwen-3-embedding/transformers/0.6b/1', torch_dtype=torch.float16)

max_length = 8192


# === Step 1: 定义 Chunk 函数 ===
def split_into_chunks(text: str, chunk_size: int = 3) -> List[str]:
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return [
        ' '.join(sentences[i:i+chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a text, retrieve similar text.'

# === Step 2: last_token_pool ===
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# === Step 3: 批量计算相似度 ===
def batch_compute_similarity(queries, documents, tokenizer, model, max_length=1024):
    input_texts = queries + documents

    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T)
    return scores.cpu().numpy()

# === Step 4: 读取数据并两两比对 ===
def process_and_compare(df, model, tokenizer):
    all_rows = []

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='Extract embedding'):
        contents = row['matched_sentences']  # e.g. [{"src1": "内容"}, {"src2": "内容2"}]

        if len(contents) < 2:
            continue
        
        for i in range(len(contents) - 1):
            c1 = contents[i]
            key1, v1 = list(c1.keys())[0], list(c1.values())[0]
            chunks1 = split_into_chunks(v1, chunk_size=3)
            queries = [
                get_detailed_instruct(task, c1.replace('.', '').strip()) for c1 in chunks1
            ]
            
            for j in range(i + 1, len(contents)):
                c2 = contents[j]
                key2, v2 = list(c2.keys())[0], list(c2.values())[0]
                if key1 == key2:
                    continue
                chunks2 = split_into_chunks(v2, chunk_size=3)

                scores = batch_compute_similarity(queries, chunks2, tokenizer, model)
                for m in range(len(queries)):
                    for n in range(len(chunks2)):
                        all_rows.append({
                            "module": row['module'],
                            "rule_id": row['rule_id'],
                            "material_id": row['material_id'],
                            "source_1": key1,
                            "source_2": key2,
                            "chunk_1": chunks1[m],
                            "chunk_2": chunks2[n],
                            "score": float(scores[m][n])
                        })

    return pd.DataFrame(all_rows)

# === Step 5: 初始化模型 ===
model.eval().cuda()


output_df = process_and_compare(external_info, model,tokenizer)

# === Step 7: 保存为 JSONL ===
output_df.to_json("pairwise_chunk_scores.jsonl", orient="records", lines=True, force_ascii=False)



!python get_sentence_match_pair.py


import pandas as pd

pair_sentence_info = pd.read_json("pairwise_chunk_scores.jsonl", lines=True)
pair_sentence_info = pair_sentence_info[pair_sentence_info['score'] >= 0.4]
pair_sentence_info.head()


pair_sentence_info.info()


# %%writefile run_vllm.py

import vllm
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List
import torch
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
import re

model_path = "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def preprocess_text(x):
    x = re.sub("http\w+", '',x)   # Delete URL
    x = re.sub(r"\.+", ".", x)    # Replace consecutive commas and periods with one comma and period character
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r"\\\(", " ", x)
    x = re.sub(r"\\\)", " ", x)
    x = re.sub(r"[ ]{1,}", " ", x)
    x = x.strip()                 # Remove empty characters at the beginning and end
    return x




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


PROMPT  = """你是一个专业的保险行业的信息处理专家，请对给定两段保险相关文本进行分析，判断两个文本所涉及到的{module}定义是否有冲突(即相同条件的表达存在不一致)，只需要回答是或者不是。
            
{module}解释: {expanation}

文本一: {content1}

文本二: {content2}

注意: 冲突是在两段文本一定相似的情况下出现了不一致，例如：
‘（六）  被保险人感染艾滋病病毒（HIV）或患艾滋病（AIDS）、高血压III级期间；’ 和 ‘（六）被保险人感染艾滋病病毒（HIV）或患艾滋病（AIDS）期间；’

请直接给出你的答案。

"""


def apply_template(row, tokenizer):
    expanation = {
        '基础产品销售信息': '该保险产品的基础配置信息，包括产品名、附加的条款信息、销售限制等',
        '投保条款': '约定该产品的保险责任细节，如保障范围、保险金额、增值服务等',
        '与保障相关的时间': '约定该产品的各类时间信息，包括但不限于犹豫期、等待期、宽限期等',
        '保障相关时间': '约定该产品的各类时间信息，包括但不限于犹豫期、等待期、宽限期等',
        '赔付 & 领取规则': '约定该产品的保险责任的赔付、给付、领取及免赔细节，如赔付年龄/比例/次数等',
        '责任免除': '约定该产品不承担保险责任的情形',
        '续保条款': '约定续保相关信息，包括但不限于续保条件、保证续保等',
        '退保条款': '约定退保相关信息，包括但不限于退保条件、退保手续费等',
        '出险条款': '约定出险相关信息，包括但不限于出险地点、出险方式等',
        '附加条款': '约定该产品的附加条款，如特别约定等',
        '术语解释': '约定该产品的术语解释，如名词定义等'
    }
    messages = [
        {
            "role": "user", 
            "content": preprocess_text(
                PROMPT.format(
                    expanation=expanation[row.module],
                    module=row.module,
                    content1=row.chunk_1,
                    content2=row.chunk_2,
                )
            )
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


# res = []
# for idx, row in pair_sentence_info.iterrows():

#     if (row['source_1'] == 'ADDITIONAL_AGREEMENT' and  row['source_2'] == 'CLAUSE') or \
#         (row['source_2'] == 'ADDITIONAL_AGREEMENT' and  row['source_1'] == 'CLAUSE'):
#         res.append(0)
#         continue
     # 对比两两句子结果
    
pair_sentence_info["text"] = pair_sentence_info.apply(lambda row: apply_template(row, tokenizer), axis=1)

print("Example:")
print(pair_sentence_info["text"].values[0])
print()

responses = llm.generate(
    pair_sentence_info["text"].values,
    vllm.SamplingParams(
        n=1,  # Number of output sequences to return for each prompt.
        top_k=1,  # Float that controls the cumulative probability of the top tokens to consider.
        temperature=0,  # randomness of the sampling
        seed=777, # Seed for reprodicibility
        skip_special_tokens=False,  # Whether to skip special tokens in the output.
        max_tokens=1,  # Maximum number of tokens to generate per output sequence.
        logits_processors=[MultipleChoiceLogitsProcessor(tokenizer, choices=["Yes", "No"])]
    ),
    use_tqdm=True
)

responses = [1 if x.outputs[0].text == 'Yes' else 0 for x in responses]
print(responses)


pair_sentence_info['res'] = responses
pair_sentence_info


pair_sentence_info.to_csv('./results.csv', index=False)


df_mean = pair_sentence_info.groupby(['material_id', 'rule_id'])['res'].mean().reset_index()


df_mean


first_submit = pd.read_json("/kaggle/input/tianchi-submit-2/submit (2).jsonl", lines=True)
first_submit


# 合并两个表
df_updated = first_submit.merge(df_mean, on=['material_id', 'rule_id'], how='left')
df_updated


df_updated['result'] = df_updated['res'].apply(lambda x: True if x < 0.05 else False)


df_updated[df_updated['result'] == True]


# df_updated
df_updated[['material_id', 'rule_id', 'result']].to_json('submit.jsonl', orient='records', lines=True)




