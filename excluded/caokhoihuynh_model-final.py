%%capture
!pip install --no-index --find-links=/kaggle/input/eedi-libraries-dataset autoawq bitsandbytes==0.45.0 peft==0.14.0 vllm==0.5.3.post1 logits-processor-zoo==0.1.0 triton


import json
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import Tensor

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from vllm import LLM, SamplingParams
from peft import LoraConfig, PeftModel, get_peft_model 
from logits_processor_zoo.vllm import GenLengthLogitsProcessor, CiteFromPromptLogitsProcessor, ForceLastPhraseLogitsProcessor, MultipleChoiceLogitsProcessor

from sklearn.neighbors import NearestNeighbors

from tqdm.auto import tqdm


%%writefile utils.py
import pandas as pd

def make_complete_query(row: pd.Series) -> str:
   
    template = "SUBJECT: {}\n\nCONSTRUCT: {}\n\nQUESTION: {}\n\nCORRECT ANSWER: {}\n\nWRONG ANSWER: {}"
    return template.format(
                            row["SubjectName"],
                            row["ConstructName"],
                            row["QuestionText"],
                            row["CorrectText"],
                            row["WrongText"],
                        )


def preprocessing_df(df: pd.DataFrame) -> pd.DataFrame:
    
    # 1. Tạo cột CorrectText
    df = df.copy()
    df = df.rename(columns={"CorrectAnswer": "CorrectChoice"})
    df["CorrectText"] = df.apply(lambda x: x[f"Answer{x['CorrectChoice']}Text"], axis=1)
    
    # 2. Tiến hành phân tách các câu trả lời thành từng dòng khác nhau
    df_melted_ans = pd.melt(
        df,
        id_vars=[  # what column to keep
            "QuestionId",
            "ConstructId",
            "ConstructName",
            "SubjectId",
            "SubjectName",
            "CorrectChoice",
            "CorrectText",
            "QuestionText",
        ],
        value_vars=[ 
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "AnswerDText",
        ],
        var_name="WrongChoice", 
        value_name="WrongText",  
    )

    # Chuẩn hóa định dạng câu trả lời
    df_melted_ans["WrongChoice"] = df_melted_ans["WrongChoice"].str[6]
    df_melted_ans = df_melted_ans.sort_values(["QuestionId", "WrongChoice"])
    df_melted_ans = df_melted_ans.reset_index(drop=True)
    
    try:
        # 3. Tách các misconceptions (chỉ dùng cho tập Train)
        df_melted_mis = pd.melt(
            df,
            id_vars=["QuestionId"],
            value_vars=[
                "MisconceptionAId",
                "MisconceptionBId",
                "MisconceptionCId",
                "MisconceptionDId",
            ],
            var_name="_melted_mis_header",
            value_name="MisconceptionId",
        )
        df_melted_mis = df_melted_mis.sort_values(["QuestionId", "_melted_mis_header"])
        df_melted_mis = df_melted_mis.drop(columns=["QuestionId", "_melted_mis_header"])
        df_melted_mis = df_melted_mis.reset_index(drop=True)
        
        assert len(df_melted_ans) == len(df_melted_mis)
        df_preprocessed = pd.concat([df_melted_ans, df_melted_mis], axis=1)
        
    except KeyError:
        df_preprocessed = df_melted_ans
        
    # 5. Loại bỏ các dòng dữ liệu của đáp án đúng, ko cần thiết cho việc dự đoán misconception
    df_preprocessed = df_preprocessed[(df_preprocessed["WrongChoice"] != df_preprocessed["CorrectChoice"])]
    try:
        df_preprocessede = df_preprocessed[df_preprocessed["MisconceptionId"].notna()]
        df_preprocessed["MisconceptionId"] = df_preprocessed["MisconceptionId"].astype(int)
    except KeyError:
        pass
    df_preprocessed = df_preprocessed.reset_index(drop=True)
    df_preprocessed["QuestionId_Answer"] = (
        df_preprocessed["QuestionId"].astype(str) + "_" + df_preprocessed["WrongChoice"]
    )
    return df_preprocessed


%%writefile infer_top25_ensemble.py
# Tạo một file Python để chạy inference trên mô hình.
# Bao gồm việc tải dữ liệu, sử dụng mô hình LoRA, và xử lý logits.

import gc
import json
import numpy as np
import pandas as pd

import torch
from torch import Tensor
import torch.nn.functional as F

from peft import LoraConfig, PeftModel, get_peft_model

from sklearn.neighbors import NearestNeighbors

from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from utils import  preprocessing_df


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

def get_embeddings_in_batches( 
                                model,
                                tokenizer: PreTrainedTokenizerBase,
                                texts: list[str],
                                max_length: int,
                                batch_size: int,
                                desc: str,
                            ):
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i : i + batch_size]
        batch_dict = tokenizer( batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad(), torch.autocast("cuda"):
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],  # type: ignore
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
            
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

def template(row: pd.Series) -> str:
    template = """Instruct: Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.
Query: 
### SubjectName: {subject}
### ConstructName: {construct}
### Question: {question}
### Correct Answer: {correct}
### Misconcepte Incorrect answer: {wrong}
<response>"""
    return template.format(
        question=row["QuestionText"],
        subject=row["SubjectName"],
        construct=row["ConstructName"],
        correct=row["CorrectText"],
        wrong=row["WrongText"],
    )


def retrieval_flow( df_test: pd.DataFrame, df_mis: pd.DataFrame, model_path: str, lora_path: str, tokenizer_path: str,) -> tuple[Tensor, Tensor]:
    
    # 1. Xử lý dữ liệu để thu được tập query và misconception
    queries = df_test.apply(template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    
    # 2. Tải model và lấy Tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, lora_path)
    
    # 3. Thực hiện trích xuất embeddings theo từng batch
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=512, batch_size=4, desc="Query embeddings"
    )
    m_embeds = get_embeddings_in_batches(
        model, tokenizer, misconceptions, max_length=512, batch_size=4, desc="Misconception embeddings"
    )
    return q_embeds, m_embeds


def main():
 
    df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
    df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
    df_test =  preprocessing_df(df_test)

    all_q_embeds = []
    all_m_embeds = []

    q_embeds, m_embeds = retrieval_flow(
       df_test,
       df_mis,
       model_path="/kaggle/input/qwen2.5-14/pytorch/default/1",
       lora_path="/kaggle/input/14b-cp750/pytorch/default/1/checkpoint-750",
       tokenizer_path="/kaggle/input/14b-cp750/pytorch/default/1/checkpoint-750",
    )
    all_q_embeds.append(q_embeds)
    all_m_embeds.append(m_embeds)
    gc.collect()
    torch.cuda.empty_cache()

   
    all_q_embeds = np.concatenate(all_q_embeds, axis=-1)
    all_m_embeds = np.concatenate(all_m_embeds, axis=-1)

    # calc
    nn = NearestNeighbors(n_neighbors=25, algorithm="brute", metric="cosine")
    nn.fit(all_m_embeds)
    dist, topk_mis = nn.kneighbors(all_q_embeds)

    # save
    savepath = "top25_miscons.json"
    with open(savepath, "w") as f:
        json.dump(topk_mis.tolist(), f)
    print(f"saved to {savepath}")


if __name__ == "__main__":
    main()


!python infer_top25_ensemble.py


import json
import time
import numpy as np
import pandas as pd


import torch

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


from sklearn.neighbors import NearestNeighbors
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from vllm import LLM, SamplingParams
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor

from utils import make_complete_query, preprocessing_df


# Bước 1: Tải và chuẩn bị dữ liệu
# Tải và xử lý dữ liệu misconception từ file CSV.
df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
orig_mis = df_mis["MisconceptionName"].tolist()

# Tải và tiền xử lý dữ liệu test từ file CSV.
df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
df_test = preprocessing_df(df_test)

# Tạo câu truy vấn đầy đủ .
df_test["QuestionComplete"] = df_test.apply(make_complete_query, axis=1)

# # Thêm danh sách top 25 misconception gần nhất cho mỗi câu hỏi từ file "top25_miscons.json"
with open("/kaggle/working/top25_miscons.json", "r") as f:
    top25_miscons = json.load(f)
    
df_test["Top25Miscons"] = top25_miscons


# Bước 2: Chuẩn bị mô hình Qwen 2.5 32B
# Khởi tạo mô hình LLM và lấy tokenizer
llm = LLM(
    "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1",
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=5120,
    disable_log_stats=True,
)
tokenizer = llm.get_tokenizer()


def generate_numbering_seq(k: int, kind: Literal["number", "alphabet"]) -> list[str]:
    if kind == "number":
        return [str(i) for i in range(1, k + 1)]
    elif kind == "alphabet":
        return [chr(ord("A")+i) for i in range(k)]
    assert False



# Khởi tạo số lượng misconceptions đầu tiên cần sắp xếp
RERANK = 9

# Tạo prompt để gửi tới mô hình LLM.
def make_llm_prompt(
    row: pd.Series,
    k: int,
    orig_mis: list[str],
) -> str:
    question = row["QuestionComplete"]
    top25_mis: list[int] = row["Top25Miscons"] 
    template = "You are an elite mathematics teacher tasked to assess the student's understanding of math concepts. Below, you will be presented with: the math question, the correct answer, the wrong answer and {k} possible misconceptions that could have led to the mistake.\n\n{question}\n\nPossible Misconceptions\n{choices}\n\nSelect one misconception that leads to incorrect answer. Just output a single number of your choice and nothing else.\n\nAnswer: "
    numbered_mis_texts = []
    for i, iseq in enumerate(generate_numbering_seq(k, "number")):
        numbered_mis_texts.append(f"{iseq}. {orig_mis[top25_mis[i]]}")
    numbered_mis_texts = "\n".join(numbered_mis_texts)
    llm_prompt = template.format(k=k, question=question, choices=numbered_mis_texts)
    return llm_prompt


df_test["PromptEn"] = df_test.apply(
    lambda row: make_llm_prompt(row, 9, orig_mis), axis=1
)


# Bước 3: thực hiện quá trình sắp xếp dựa trên prompt và Multiple Choice Logits Processor
# Tạo một bộ xử lý logits cho các câu hỏi trắc nghiệm (multiple choice)
logits_processor = MultipleChoiceLogitsProcessor(
    tokenizer=tokenizer,
    choices=generate_numbering_seq(RERANK, "number")
)

# Thiết lập các tham số cho sampling (sinh mẫu) khi gọi mô hình
sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=1,
    logits_processors=[logits_processor],
    logprobs=RERANK,
)
# Gọi mô hình LLM để sinh ra các phản hồi dựa trên các prompts.
responses = llm.generate(df_test["PromptEn"].tolist(), sampling_params)



# Thực hiện bước re-ranking các misconceptions dựa trên dự đoán từ mô hình LLM
all_reranked = []
for resp, top25 in zip(responses, df_test["Top25Miscons"]):
    decoded_tokens = [logprob.decoded_token for logprob in resp.outputs[0].logprobs[0].values()]

    # Chuyển các giá trị token về giá trị hệ số.
    indices = [int(d) - 1 for d in decoded_tokens]
    
    # Sắp xếp lại top RERANK các misconception đầu
    reranked = np.array(top25[:RERANK])[indices].tolist() + top25[RERANK:]
    all_reranked.append(reranked)
    
assert len(all_reranked) == df_test.shape[0]


# Bước 4: Tạo file submission
# File chứa hai cột chính: cột ID question và cột ID Misconception
df_test["MisconceptionId"] = [" ".join(str(x) for x in row) for row in all_reranked]

df_sub = df_test[["QuestionId_Answer", "MisconceptionId"]]

df_sub.to_csv("submission.csv", index=False)


pd.read_csv("submission.csv")

