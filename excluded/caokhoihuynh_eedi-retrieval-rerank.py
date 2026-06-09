%%capture
!pip install --no-index --find-links=/kaggle/input/eedi-libraries-dataset autoawq bitsandbytes==0.45.0 peft==0.14.0 vllm==0.5.3.post1 logits-processor-zoo==0.1.0 triton


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


def make_nice_df(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a valid train or test dataframe. Valid means the misconception id must be not null
    and the dataset is melted to ease row-by-row inference. For train, we are need to melt 2 times
    because we want to melt both answer and miconceptions in unison.
    """
    # 1. duplicate correct answer text to its own column
    df = df.copy()
    df = df.rename(columns={"CorrectAnswer": "CorrectChoice"})
    df["CorrectText"] = df.apply(lambda x: x[f"Answer{x['CorrectChoice']}Text"], axis=1)
    # 2. melt answers
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
        value_vars=[  # what columns to transform to rows (melted)
            "AnswerAText",
            "AnswerBText",
            "AnswerCText",
            "AnswerDText",
        ],
        var_name="WrongChoice",  # rename the column that holds melted-column's headers
        value_name="WrongText",  # rename the column that holds melted-column's content
    )
    df_melted_ans["WrongChoice"] = df_melted_ans["WrongChoice"].str[6]
    df_melted_ans = df_melted_ans.sort_values(["QuestionId", "WrongChoice"])
    df_melted_ans = df_melted_ans.reset_index(drop=True)
    try:
        # 3. melt misconceptions (only available at train dataset)
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
        # 4. combine
        assert len(df_melted_ans) == len(df_melted_mis)
        df_nice = pd.concat([df_melted_ans, df_melted_mis], axis=1)
    except KeyError:
        # test set does not have misconceptions
        df_nice = df_melted_ans
    # 5. clean
    df_nice = df_nice[(df_nice["WrongChoice"] != df_nice["CorrectChoice"])]
    try:
        df_nice = df_nice[df_nice["MisconceptionId"].notna()]
        df_nice["MisconceptionId"] = df_nice["MisconceptionId"].astype(int)
    except KeyError:
        pass
    df_nice = df_nice.reset_index(drop=True)
    df_nice["QuestionId_Answer"] = (
        df_nice["QuestionId"].astype(str) + "_" + df_nice["WrongChoice"]
    )
    return df_nice


%%writefile infer_top25_ensemble.py
import gc
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from utils import make_nice_df


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    # copied from: https://huggingface.co/Salesforce/SFR-Embedding-2_R
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
        batch_dict = tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            outputs = model(**batch_dict)
            batch_embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],  # type: ignore
            )
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1).cpu()
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


def anhvth226_template(row: pd.Series) -> str:
    template = """<instruct>Given a math multiple-choice problem with a student's wrong answer, retrieve the math misconceptions
<query>Question: {question}
    
SubjectName: {subject}
ConstructName: {construct}
Correct answer: {correct}
Student wrong answer: {wrong}
<response>"""
    return template.format(
        question=row["QuestionText"],
        subject=row["SubjectName"],
        construct=row["ConstructName"],
        correct=row["CorrectText"],
        wrong=row["WrongText"],
    )


def anhvth226_flow(
    df_test: pd.DataFrame,
    df_mis: pd.DataFrame,
    model_path: str,
    lora_path: str,
    tokenizer_path: str,
) -> tuple[Tensor, Tensor]:
    # load dataset
    queries = df_test.apply(anhvth226_template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    # load model and tokenizer
    # i modifiy a bit because i cant load the awq
    # model = AutoModel.from_pretrained(
    #     model_path, device_map=0, torch_dtype=torch.float16, load_in_4bit=False
    # )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
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
    # batch infer both
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=320, batch_size=4, desc="anhvth226 Q"
    )
    m_embeds = get_embeddings_in_batches(
        model,
        tokenizer,
        misconceptions,
        max_length=320,
        batch_size=4,
        desc="anhvth226 M",
    )
    return q_embeds, m_embeds


def mschoo_template(row: pd.Series) -> str:
    template = """Instruct: Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.
Query: ### SubjectName: {subject}
### ConstructName: {subject}
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


def mschoo_flow(
    df_test: pd.DataFrame,
    df_mis: pd.DataFrame,
    model_path: str,
    lora_path: str,
    tokenizer_path: str,
) -> tuple[Tensor, Tensor]:
    # load dataset
    queries = df_test.apply(mschoo_template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    # load model and tokenizer
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
    # batch infer both
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=512, batch_size=4, desc="mschoo Q"
    )
    m_embeds = get_embeddings_in_batches(
        model, tokenizer, misconceptions, max_length=512, batch_size=4, desc="mschoo M"
    )
    return q_embeds, m_embeds


def zuoyouzuo_template(row: pd.Series) -> str:
    template = """Instruct: Given a math question with correct answer and a misconcepted incorrect answer, retrieve the most accurate misconception for the incorrect answer.
Query: ### SubjectName: {subject}
### ConstructName: {subject}
### Question: {question}
### Correct Answer: {correct}
### Misconcepte Incorrect answer: {wrong}"""
    return template.format(
        question=row["QuestionText"],
        subject=row["SubjectName"],
        construct=row["ConstructName"],
        correct=row["CorrectText"],
        wrong=row["WrongText"],
    )


def zuoyouzuo_flow(
    df_test: pd.DataFrame,
    df_mis: pd.DataFrame,
    model_path: str,
    lora_path: str,
    tokenizer_path: str,
) -> tuple[Tensor, Tensor]:
    # load dataset
    queries = df_test.apply(zuoyouzuo_template, axis=1).tolist()
    misconceptions = df_mis["MisconceptionName"].tolist()
    # load model and tokenizer
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
    config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, config)
    d = torch.load(lora_path, map_location=model.device)
    model.load_state_dict(d, strict=False)
    model = model.merge_and_unload()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # batch infer both
    q_embeds = get_embeddings_in_batches(
        model, tokenizer, queries, max_length=512, batch_size=4, desc="zuoyouzuo Q"
    )
    m_embeds = get_embeddings_in_batches(
        model,
        tokenizer,
        misconceptions,
        max_length=512,
        batch_size=4,
        desc="zuoyouzuo M",
    )
    return q_embeds, m_embeds


def main():
    # load dataset
    df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
    df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
    df_test = make_nice_df(df_test)

    all_q_embeds = []
    all_m_embeds = []

    # (1) anhvth226
    q_embeds1, m_embeds1 = anhvth226_flow(
        df_test,
        df_mis,
        model_path="/kaggle/input/qwen2.5-14/pytorch/default/1",
        lora_path="/kaggle/input/2211-lora-14b/transformers/default/1",
        tokenizer_path="/kaggle/input/2211-lora-14b/transformers/default/1",
    )
    all_q_embeds.append(q_embeds1)
    all_m_embeds.append(m_embeds1)
    gc.collect()
    torch.cuda.empty_cache()

    # # (2) mschoo
    # q_embeds2, m_embeds2 = mschoo_flow(
    #    df_test,
    #    df_mis,
    #    model_path="/kaggle/input/qwen2.5-14/pytorch/default/1",
    #    lora_path="/kaggle/input/14b-cp750/pytorch/default/1/checkpoint-750",
    #    tokenizer_path="/kaggle/input/14b-cp750/pytorch/default/1/checkpoint-750",
    # )
    # all_q_embeds.append(q_embeds2)
    # all_m_embeds.append(m_embeds2)
    # gc.collect()
    # torch.cuda.empty_cache()

    # # (3) zuoyouzuo
    # q_embeds3, m_embeds3 = zuoyouzuo_flow(
    #    df_test,
    #    df_mis,
    #    model_path="/kaggle/input/qwen2.5-14/pytorch/default/1",
    #    lora_path="/kaggle/input/qwen14b-it-lora/lora_weights/adapter.bin",
    #    tokenizer_path="/kaggle/input/qwen14b-it-lora/lora_weights",
    # )
    # all_q_embeds.append(q_embeds3)
    # all_m_embeds.append(m_embeds3)
    # gc.collect()
    # torch.cuda.empty_cache()

    # # concat sideways
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


%%writefile rerank_qwen14b_model.py

import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from typing import Literal

from utils import make_complete_query, make_nice_df


from vllm import LLM, SamplingParams
from logits_processor_zoo.vllm import GenLengthLogitsProcessor, CiteFromPromptLogitsProcessor, ForceLastPhraseLogitsProcessor, MultipleChoiceLogitsProcessor


# load dataset
df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
orig_mis = df_mis["MisconceptionName"].tolist()

df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
df_test = make_nice_df(df_test)
df_test["QuestionComplete"] = df_test.apply(make_complete_query, axis=1)
with open("/kaggle/working/top25_miscons.json", "r") as f:
    top25_miscons = json.load(f)
    
df_test["Top25Miscons"] = top25_miscons


llm = LLM(
    "/kaggle/input/qwenqwen2-5-14b-instruct-awq/Qwen2.5-14B-Instruct-AWQ",
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=1024,
    disable_log_stats=True,
)
tokenizer = llm.get_tokenizer()


def generate_numbering_seq(k: int, kind: Literal["number", "alphabet"]) -> list[str]:
    if kind == "number":
        return [str(i) for i in range(1, k + 1)]
    elif kind == "alphabet":
        return [chr(ord("A")+i) for i in range(k)]
    assert False


RERANK = 9

def make_llm_prompt_en(
    row: pd.Series,
    k: int,
    orig_mis: list[str],
) -> str:
    question = row["QuestionComplete"]
    top25_mis: list[int] = row["Top25Miscons"]  # type: ignore
    template = "You are an elite mathematics teacher tasked to assess the student's understanding of math concepts. Below, you will be presented with: the math question, the correct answer, the wrong answer and {k} possible misconceptions that could have led to the mistake.\n\n{question}\n\nPossible Misconceptions\n{choices}\n\nSelect one misconception that leads to incorrect answer. Just output a single number of your choice and nothing else.\n\nAnswer: "
    numbered_mis_texts = []
    for i, iseq in enumerate(generate_numbering_seq(k, "number")):
        numbered_mis_texts.append(f"{iseq}. {orig_mis[top25_mis[i]]}")
    numbered_mis_texts = "\n".join(numbered_mis_texts)
    llm_prompt = template.format(k=k, question=question, choices=numbered_mis_texts)
    return llm_prompt

df_test["PromptEn"] = df_test.apply(
    lambda row: make_llm_prompt_en(row, RERANK, orig_mis), axis=1
)

# english
logits_processor = MultipleChoiceLogitsProcessor(
    tokenizer=tokenizer,
    choices=generate_numbering_seq(RERANK, "number")
)
sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=1,
    logits_processors=[logits_processor],
    logprobs=RERANK,
)
# responses_en = llm.generate(df_test["PromptEn"].tolist(), sampling_params)


# all_reranked = []
# for resp, top25 in zip(responses_en, df_test["Top25Miscons"]):
#     decoded_tokens = [logprob.decoded_token for logprob in resp.outputs[0].logprobs[0].values()]

#     # map back to 0-based int
#     indices = [int(d) - 1 for d in decoded_tokens]
#     # rerank the first 9 items from 25
#     reranked = np.array(top25[:RERANK])[indices].tolist() + top25[RERANK:]
#     all_reranked.append(reranked)
# assert len(all_reranked) == df_test.shape[0]
all_reranked = top25_miscons

df_test["MisconceptionId"] = [" ".join(str(x) for x in row) for row in all_reranked]

df_sub = df_test[["QuestionId_Answer", "MisconceptionId"]]

df_sub.to_csv("submission.csv", index=False)


!python rerank_qwen14b_model.py


%%writefile rerank_qwen32b_model.py

import json
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from typing import Literal

from utils import make_complete_query, make_nice_df


from vllm import LLM, SamplingParams
from logits_processor_zoo.vllm import GenLengthLogitsProcessor, CiteFromPromptLogitsProcessor, ForceLastPhraseLogitsProcessor, MultipleChoiceLogitsProcessor


# load dataset
df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
orig_mis = df_mis["MisconceptionName"].tolist()

df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
df_test = make_nice_df(df_test)
df_test["QuestionComplete"] = df_test.apply(make_complete_query, axis=1)


rerank_df = pd.read_csv("/kaggle/working/rerank_qwen14b.csv")
df_test["Top25Miscons"] = rerank_df['MisconceptionId']

df_test["Top25Miscons"] = df_test["Top25Miscons"] .apply( lambda x: [int(id_str) for id_str in x.split()])

llm = LLM(
    "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1",
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85, 
    trust_remote_code=True,
    dtype="half", 
    enforce_eager=True,
    max_model_len=1024,
    disable_log_stats=True,
)
tokenizer = llm.get_tokenizer()


def generate_numbering_seq(k: int, kind: Literal["number", "alphabet"]) -> list[str]:
    if kind == "number":
        return [str(i) for i in range(1, k + 1)]
    elif kind == "alphabet":
        return [chr(ord("A")+i) for i in range(k)]
    assert False


RERANK = 5

def make_llm_prompt_en(
    row: pd.Series,
    k: int,
    orig_mis: list[str],
) -> str:
    question = row["QuestionComplete"]
    top25_mis: list[int] = row["Top25Miscons"]  # type: ignore
    template = "You are an elite mathematics teacher tasked to assess the student's understanding of math concepts. Below, you will be presented with: the math question, the correct answer, the wrong answer and {k} possible misconceptions that could have led to the mistake.\n\n{question}\n\nPossible Misconceptions\n{choices}\n\nSelect one misconception that leads to incorrect answer. Just output a single number of your choice and nothing else.\n\nAnswer: "
    numbered_mis_texts = []
    for i, iseq in enumerate(generate_numbering_seq(k, "number")):
        numbered_mis_texts.append(f"{iseq}. {orig_mis[top25_mis[i]]}")
    numbered_mis_texts = "\n".join(numbered_mis_texts)
    llm_prompt = template.format(k=k, question=question, choices=numbered_mis_texts)
    return llm_prompt

df_test["PromptEn"] = df_test.apply(
    lambda row: make_llm_prompt_en(row, 5, orig_mis), axis=1
)

# english
logits_processor = MultipleChoiceLogitsProcessor(
    tokenizer=tokenizer,
    choices=generate_numbering_seq(RERANK, "number")
)
sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=1,
    logits_processors=[logits_processor],
    logprobs=RERANK,
)
responses_en = llm.generate(df_test["PromptEn"].tolist(), sampling_params)


all_reranked = []
for resp, top25 in zip(responses_en, df_test["Top25Miscons"]):
    decoded_tokens = [logprob.decoded_token for logprob in resp.outputs[0].logprobs[0].values()]

    # map back to 0-based int
    indices = [int(d) - 1 for d in decoded_tokens]
    # rerank the first 9 items from 25
    reranked = np.array(top25[:RERANK])[indices].tolist() + top25[RERANK:]
    all_reranked.append(reranked)
assert len(all_reranked) == df_test.shape[0]
df_test["MisconceptionId"] = [" ".join(str(x) for x in row) for row in all_reranked]

df_sub = df_test[["QuestionId_Answer", "MisconceptionId"]]

df_sub.to_csv("submission.csv", index=False)


# !python rerank_qwen32b_model.py


# %%writefile rerank_qwen32b_model_top1.py

# import json
# import time
# from argparse import ArgumentParser
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Literal

# import numpy as np
# import pandas as pd
# import torch
# from sklearn.neighbors import NearestNeighbors
# from transformers import (
#     AutoModel,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     PreTrainedModel,
#     PreTrainedTokenizerBase,
# )

# from typing import Literal

# from utils import make_complete_query, make_nice_df


# from vllm import LLM, SamplingParams
# from logits_processor_zoo.vllm import GenLengthLogitsProcessor, CiteFromPromptLogitsProcessor, ForceLastPhraseLogitsProcessor, MultipleChoiceLogitsProcessor


# # load dataset
# df_mis = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
# orig_mis = df_mis["MisconceptionName"].tolist()

# df_test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
# df_test = make_nice_df(df_test)
# df_test["QuestionComplete"] = df_test.apply(make_complete_query, axis=1)


# rerank_df = pd.read_csv("/kaggle/working/rerank_qwen14b_top_5.csv")
# df_test["Top25Miscons"] = rerank_df['MisconceptionId']

# df_test["Top25Miscons"] = df_test["Top25Miscons"] .apply( lambda x: [int(id_str) for id_str in x.split()])

# llm = LLM(
#     "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1",
#     quantization="awq",
#     tensor_parallel_size=2,
#     gpu_memory_utilization=0.85, 
#     trust_remote_code=True,
#     dtype="half", 
#     enforce_eager=True,
#     max_model_len=1024,
#     disable_log_stats=True,
# )
# tokenizer = llm.get_tokenizer()


# def generate_numbering_seq(k: int, kind: Literal["number", "alphabet"]) -> list[str]:
#     if kind == "number":
#         return [str(i) for i in range(1, k + 1)]
#     elif kind == "alphabet":
#         return [chr(ord("A")+i) for i in range(k)]
#     assert False


# RERANK = 1

# def make_llm_prompt_en(
#     row: pd.Series,
#     k: int,
#     orig_mis: list[str],
# ) -> str:
#     question = row["QuestionComplete"]
#     top25_mis: list[int] = row["Top25Miscons"]  # type: ignore
#     template = "You are an elite mathematics teacher tasked to assess the student's understanding of math concepts. Below, you will be presented with: the math question, the correct answer, the wrong answer and {k} possible misconceptions that could have led to the mistake.\n\n{question}\n\nPossible Misconceptions\n{choices}\n\nSelect one misconception that leads to incorrect answer. Just output a single number of your choice and nothing else.\n\nAnswer: "
#     numbered_mis_texts = []
#     for i, iseq in enumerate(generate_numbering_seq(k, "number")):
#         numbered_mis_texts.append(f"{iseq}. {orig_mis[top25_mis[i]]}")
#     numbered_mis_texts = "\n".join(numbered_mis_texts)
#     llm_prompt = template.format(k=k, question=question, choices=numbered_mis_texts)
#     return llm_prompt

# df_test["PromptEn"] = df_test.apply(
#     lambda row: make_llm_prompt_en(row, 5, orig_mis), axis=1
# )

# # english
# logits_processor = MultipleChoiceLogitsProcessor(
#     tokenizer=tokenizer,
#     choices=generate_numbering_seq(RERANK, "number")
# )
# sampling_params = SamplingParams(
#     n=1,
#     temperature=0,
#     max_tokens=1,
#     logits_processors=[logits_processor],
#     logprobs=RERANK,
# )
# responses_en = llm.generate(df_test["PromptEn"].tolist(), sampling_params)


# all_reranked = []
# for resp, top25 in zip(responses_en, df_test["Top25Miscons"]):
#     decoded_tokens = [logprob.decoded_token for logprob in resp.outputs[0].logprobs[0].values()]

#     # map back to 0-based int
#     indices = [int(d) - 1 for d in decoded_tokens]
#     # rerank the first 9 items from 25
#     reranked = np.array(top25[:RERANK])[indices].tolist() + top25[RERANK:]
#     all_reranked.append(reranked)
# assert len(all_reranked) == df_test.shape[0]
# df_test["MisconceptionId"] = [" ".join(str(x) for x in row) for row in all_reranked]

# df_sub = df_test[["QuestionId_Answer", "MisconceptionId"]]

# df_sub.to_csv("submission.csv", index=False)


# !python rerank_qwen32b_model_top1.py


import pandas as pd


# # # final check

# pd.read_csv("/kaggle/working/rerank_qwen14b_top_5.csv")


# # final check

pd.read_csv("/kaggle/working/submission.csv")




