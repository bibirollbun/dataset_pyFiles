%%time
!pip install --no-index --find-links=/kaggle/input/vllm-offline-install blake3 msgspec py-cpuinfo tqdm requests transformers triton  
!pip install --no-index --find-links=/kaggle/input/vllm-offline-install vllm


!pip install -q -U /kaggle/input/hf-libraries/transformers/transformers-4.51.3-py3-none-any.whl
!pip install -q -U /kaggle/input/hf-libraries/sentence-transformers/sentence_transformers-3.4.1-py3-none-any.whl
!pip install -q -U /kaggle/input/hf-libraries/peft/peft-0.14.0-py3-none-any.whl
!pip install -q -U /kaggle/input/hf-libraries/accelerate/accelerate-1.5.2-py3-none-any.whl
!pip install -q -U /kaggle/input/logits-processor-zoo/logits_processor_zoo-0.1.2-py3-none-any.whl


%%capture
!pip install --no-index /kaggle/input/bitsandbytes0-42-0/bitsandbytes-0.42.0-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
!pip install --no-index  /kaggle/input/bitsandbytes0-42-0/optimum-1.21.2-py3-none-any.whl --find-links=/kaggle/input/bitsandbytes0-42-0
!pip install --no-index  /kaggle/input/bitsandbytes0-42-0/auto_gptq-0.7.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --find-links=/kaggle/input/bitsandbytes0-42-0


import pandas as pd
import os
IS_SUBMISSION = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
test_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
train_df = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")


import pandas as pd
from typing import List, Dict, Union

def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame to create a structured dataset for misconception analysis.
    Works with both training and test sets by handling missing misconception columns.

    Args:
        df: Input DataFrame containing answer and misconception data

    Returns:
        Processed DataFrame with restructured data
    """
    datas: List[Dict[str, Union[str, int]]] = []

    for _, row in df.iterrows():
        correct_col = row["CorrectAnswer"]
        correct_answer_col = f"Answer{correct_col}Text"

        for col in ["A", "B", "C", "D"]:
            if correct_col == col:
                continue  # Bỏ qua phương án đúng

            answer_col = f"Answer{col}Text"
            misconception_col = f"Misconception{col}Id"

            # Thử lấy misconception ID nếu có
            try:
                if pd.isna(row[misconception_col]):
                    continue
                misconception_id = int(row[misconception_col])
            except (KeyError, TypeError):
                misconception_id = None  # Với test set: không thêm

            data = {
                "Answer": row[answer_col],
                "Correct": row[correct_answer_col],
                "QuestionId_Answer": f"{row['QuestionId']}_{col}"
            }

            if misconception_id is not None:
                data["MisconceptionId"] = misconception_id

            row_dict = row.to_dict()

            # Xoá các cột không cần
            for c in ["A", "B", "C", "D"]:
                row_dict.pop(f"Misconception{c}Id", None)
                row_dict.pop(f"Answer{c}Text", None)

            data.update(row_dict)
            datas.append(data)

    return pd.DataFrame(datas)


processed_test_df = load_data(test_df)
processed_train_df= load_data(train_df)


processed_test_df.columns


processed_test_df.to_parquet("/kaggle/working/processed_test_df.parquet", index=False)


%%writefile distill.yaml
save_dir: "/kaggle/working"

# distill.py
distill:
  model_name: "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
  input_name: "processed_test_df.parquet"
  save_name: "processed_test_df_with_Qwen2.5_Math_7B_Instruct.parquet"


%%writefile distill.py
from pathlib import Path
from typing import List

import pandas as pd
import typer
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from vllm import LLM, SamplingParams
import torch
# Set random seed for reproducibility
set_seed(42)

# Template for generating prompts
PROMPT_FORMAT: str = """<|im_start|>system
You will be given a math problem and its correct and incorrect answer.
First explain why the correct answer is correct, and finally explain reasons and misconceptions for incorrect answer.
Please briefly explain in 200 words or less.<|im_end|>
<|im_start|>user
Problem: {QuestionText}\nCorrect Answer: {Correct}\nIncorrect Answer: {Answer}.<|im_end|>
<|im_start|>assistant
"""


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        DictConfig: Loaded configuration
    """
    try:
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig):
            raise ValueError(f"Config loaded from {config_path} is not a DictConfig")
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {str(e)}")


def create_model(params: DictConfig) -> LLM:
    """
    Initialize the LLM model with specified parameters.

    Args:
        params (DictConfig): Model configuration parameters

    Returns:
        LLM: Initialized model
    """
    return LLM(
        model=params.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.99,
        dtype=torch.float16,
        max_model_len=4096,
        enforce_eager=True,
        tensor_parallel_size=2
    )


def generate_prompts(df: pd.DataFrame) -> List[str]:
    """
    Generate prompts from DataFrame using the template.

    Args:
        df (pd.DataFrame): Input DataFrame containing question data

    Returns:
        List[str]: List of formatted prompts
    """
    return df.apply(lambda x: PROMPT_FORMAT.format(**x), axis=1).tolist()


def generate_responses(model: LLM, prompts: List[str]) -> List[str]:
    """
    Generate responses using the model for given prompts.

    Args:
        model (LLM): Initialized LLM model
        prompts (List[str]): List of input prompts

    Returns:
        List[str]: Generated responses
    """
    sampling_params = SamplingParams(
        max_tokens=4096,
        stop=["<|im_end|>"],
        temperature=0.0,
    )
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def main(
    config: str = "/kaggle/working/distill.yaml",
) -> None:
    """
    Main function to run the distillation process.

    Args:
        config (str): Path to configuration file
    """
    try:
        # Load configuration
        cfg = load_config(config)
        params = cfg.distill

        # Read input data
        input_path = Path(cfg.save_dir) / params.input_name
        df = pd.read_parquet(input_path)

        # Generate prompts
        df["prompt"] = generate_prompts(df)

        # Initialize model and generate responses
        model = create_model(params)
        df["kd"] = generate_responses(model, df.prompt.tolist())

        # Save results
        output_path = Path(cfg.save_dir) / params.save_name
        df.to_parquet(output_path, index=False)

    except Exception as e:
        raise RuntimeError(f"Distillation process failed: {str(e)}")


if __name__ == "__main__":
    typer.run(main)


!CUDA_VISIBLE_DEVICES=0,1 python /kaggle/working/distill.py


%%writefile biencoder.yaml
save_dir: "/kaggle/working"
input_dir: "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
inference_biencoder:
  input_name: "processed_test_df_with_Qwen2.5_Math_7B_Instruct.parquet"
  save_name: "data_bi_with_Qwen2.5_Math_7B_Instruct.parquet"
  model_name: "/kaggle/input/stella_1.5b/transformers/default/1"
  is_lora: true
  load_in_4bit: false
  batch_size: 4


%%writefile infer_biencoder.py
import gc
from pathlib import Path
from typing import List

import pandas as pd
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Prompt template for formatting the input data
PROMPT_FORMAT: str = """Subject: {SubjectName}\nConstruct: {ConstructName}\nQuestion: {QuestionText}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}"""


def load_model(
    params: DictConfig,
    cfg: DictConfig
) -> SentenceTransformer:
    model = SentenceTransformer(params.model_name)

    return model


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Encode texts using the SentenceTransformer model.

    Args:
        model: SentenceTransformer model
        texts: List of texts to encode
        batch_size: Batch size for encoding

    Returns:
        Tensor of encoded texts
    """
    return model.encode(
        texts,
        convert_to_tensor=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )


def process_df(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    model: SentenceTransformer,
    params: DictConfig
) -> pd.DataFrame:

    # Encode texts
    mapping_encoddings = encode_texts(
        model,
        mapping["MisconceptionName"].tolist(),
        params.batch_size,
    )
    prompt_encoddings = encode_texts(
        model,
        df["kd"].tolist(),
        params.batch_size,
    )

    # Calculate similarities and get predictions
    similarity = model.similarity(prompt_encoddings, mapping_encoddings).cpu().numpy()
    indices = similarity.argsort()[:, ::-1]
    df["pred_ids"] = [" ".join(map(str, idxs)) for idxs in indices]

    return df


def main(config: str = "/kaggle/working/biencoder.yaml") -> None:
    """
    Main function to run the inference process.

    Args:
        config: Path to the configuration file
    """
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.inference_biencoder

    # Load input data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_parquet(Path(cfg.save_dir) / params.input_name)

    # Format prompts
    df["prompt"] = df.apply(
        lambda x: PROMPT_FORMAT.format(**x),
        axis=1,
    )

    processed_dfs: List[pd.DataFrame] = []

    model = load_model(params, cfg)
    df = process_df(df, mapping, model, params)
    processed_dfs.append(df)

    # Clean up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save results
    pd.concat(processed_dfs).to_parquet(Path(cfg.save_dir) / params.save_name, index=False)


if __name__ == "__main__":
    typer.run(main)


!python /kaggle/working/infer_biencoder.py


%%writefile reranker.yaml
save_dir: "/kaggle/working"
input_dir: "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
inference_listwise:
  model_name: "/kaggle/input/qwen2.5-7b-eedi/transformers/default/1"
  input_name: "data_bi_with_Qwen2.5_Math_7B_Instruct.parquet"
  save_name: "data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_1.parquet"
  batch_size: 1
  seed: 42
  topk: 100
  num_choice: 50
  num_slide: 50
  add_na: True
  max_length: 2048
  load_in_4bit: false


%%writefile reranker.py
import math
import string
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from vllm import LLM, SamplingParams


# Disable progress bar for datasets
disable_progress_bar()

PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List
{mis_names}<|im_end|>
<|im_start|>assistant
"""

NA_PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options. If there are no suitable options, return NA.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List (rank: {rank})
{mis_names}<|im_end|>
<|im_start|>assistant
"""


def get_choice_words(num_choices: int) -> List[str]:
    """Generate a list of choice identifiers (A, B, C, etc.)."""
    alphabets = list(string.ascii_uppercase + string.ascii_lowercase)
    return alphabets[:num_choices]


def tokenize_function(row: Dict, tokenizer: Any, max_length: int) -> Dict:
    """Tokenize text input using the specified tokenizer."""
    embeddings = tokenizer.encode_plus(
        row["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.squeeze(0) for k, v in embeddings.items()}


def process_data(
    df: pd.DataFrame,
    tokenizer: Any,
    target_cols: List[str] = ["prompt"],
    max_length: int = 1536,
) -> Dataset:
    """Process DataFrame into a tokenized dataset."""
    dataset = Dataset.from_pandas(df[target_cols])
    return dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=max_length),
        batched=False,
        num_proc=1,
    )


@torch.no_grad()
@torch.amp.autocast("cuda")
def inference(
    df: pd.DataFrame, model: Any, target_tokens: List[int], batch_size: int, tokenizer: Any
) -> pd.DataFrame:
    """Perform model inference on the input data."""
    end_idx = 0
    logit_list = []

    for start_idx in tqdm(range(0, len(df)), total=len(df), desc="Inference"):
        if start_idx < end_idx:
            continue

        # Process batch
        end_idx = min(len(df), start_idx + batch_size)
        tmp = df.iloc[start_idx:end_idx].copy()
        dset = process_data(tmp, tokenizer)

        # Prepare inputs
        tmp["input_ids"] = dset["input_ids"]
        tmp["attention_mask"] = dset["attention_mask"]
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {
                "input_ids": tmp["input_ids"].tolist(),
                "attention_mask": tmp["attention_mask"].tolist(),
            },
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        ).to(model.device)

        # Get model outputs
        outputs = model(**inputs)
        logits = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        # Extract last token logits
        last_token_logits = []
        for logit, mask in zip(logits, inputs["attention_mask"].cpu().numpy()):
            last_token_idx = mask.nonzero()[0][-1]
            last_token_logits.append(logit[last_token_idx, target_tokens])
        logit_list.extend(last_token_logits)

    df["logit"] = logit_list
    return df


def add_prompt(df: pd.DataFrame, mapping: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Create dataset with misconception options."""
    df["pred_ids"] = df["pred_ids"].apply(lambda x: list(map(int, x.split())))
    new_rows = []

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for i in range(params.topk // params.num_slide):
            if params.num_slide * i + params.num_choice > params.topk:
                break

            new_row = row.copy()
            mis_ids = row["pred_ids"][
                params.num_slide * i : params.num_slide * i + params.num_choice
            ]
            new_row["pred_ids"] = mis_ids

            # Format misconception names
            names = mapping.loc[mis_ids, "MisconceptionName"].tolist()
            names = "\n".join(
                [f"{x}: {y}" for x, y in zip(params.choice_words[: params.num_choice], names)]
            )

            new_row["mis_names"] = names
            new_row["idx"] = idx
            start_idx = params.num_slide * i
            end_idx = params.num_slide * i + params.num_choice
            new_row["rank"] = f"{start_idx + 1}-{end_idx}"
            new_rows.append(new_row)

    # Create final DataFrame
    df = pd.DataFrame(new_rows)
    df["last_choice"] = params.choice_words[-1]
    df["prompt"] = df.apply(
        lambda x: NA_PROMPT_FORMAT.format(**x) if params.add_na else PROMPT_FORMAT.format(**x),
        axis=1,
    )
    return df


def main(
    config: str = "/kaggle/working/reranker.yaml",
) -> None:
    """Main function to run the inference pipeline."""
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.inference_listwise
    set_seed(params.seed)

    # Load data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_parquet(Path(cfg.save_dir) / params.input_name)


    # Initialize tokenizer
    model_name = params.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"

    # Prepare choice tokens
    params.choice_words = get_choice_words(params.num_choice)
    params.choice_tokens = [tokenizer.encode(x)[0] for x in params.choice_words]

    # Process dataset
    tmp_df = add_prompt(df.copy(), mapping, params)
    tmp_df["length"] = tmp_df["prompt"].apply(lambda x: len(x.split()))
    tmp_df.sort_values("length", inplace=True, ascending=False)

    # Initialize model
    model = LLM(
        model=params.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.99,
        max_logprobs=52,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
        tensor_parallel_size=2
    )

    # Generate predictions
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=params.num_choice)
    outputs = model.generate(tmp_df["prompt"].tolist(), sampling_params)

    # Process outputs
    logits = []
    for output in outputs:
        output = output.outputs[0].logprobs[0]
        score_dict = {i: 0.0 for i in range(52)}
        for k in output.keys():
            if k in params.choice_tokens:
                score_dict[params.choice_tokens.index(k)] = math.exp(output[k].logprob)
        logits.append(list(score_dict.values()))

    tmp_df["logit"] = logits

    # Aggregate results
    new_rows = []
    for i, g in tmp_df.groupby("idx"):
        id_dict = defaultdict(list)
        for idx, row in g.iterrows():
            for pred_id, logit in zip(row["pred_ids"], row["logit"]):
                id_dict[pred_id].append(logit)
        id_dict = {k: np.mean(v) for k, v in id_dict.items()}
        sorted_ids = sorted(id_dict, key=id_dict.get, reverse=True)
        row = g.iloc[0].copy()
        row["pred_ids"] = sorted_ids
        new_rows.append(row)

    tmp_df = pd.DataFrame(new_rows)
    tmp_df["pred_ids"] = tmp_df["pred_ids"].apply(lambda x: " ".join(map(str, x)))
    try:
        tmp_df.to_parquet(Path(cfg.save_dir) / params.save_name, index=False)
    except Exception as e:
        print(e)
if __name__ == "__main__":
    typer.run(main)


!python /kaggle/working/reranker.py


%%writefile reranker2.yaml
save_dir: "/kaggle/working"
input_dir: "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
inference_listwise:
  model_name: "/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1"
  input_name: "data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_1.parquet"
  save_name: "data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_2.parquet"
  batch_size: 1
  seed: 42
  topk: 50
  num_choice: 25
  num_slide: 25
  add_na: True
  max_length: 2048
  load_in_4bit: false


%%writefile reranker2.py
import math
import string
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import typer
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from vllm import LLM, SamplingParams
import torch

# Disable progress bar for datasets
disable_progress_bar()

PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List
{mis_names}<|im_end|>
<|im_start|>assistant
"""

NA_PROMPT_FORMAT: str = """<|im_start|>system
You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
Please return the most appropriate option from the list of misconceptions. Do not output anything other than options. If there are no suitable options, return NA.<|im_end|>
<|im_start|>user
# Math Problem
Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# Misconception List (rank: {rank})
{mis_names}<|im_end|>
<|im_start|>assistant
"""


def get_choice_words(num_choices: int) -> List[str]:
    """Generate a list of choice identifiers (A, B, C, etc.)."""
    alphabets = list(string.ascii_uppercase + string.ascii_lowercase)
    return alphabets[:num_choices]


def tokenize_function(row: Dict, tokenizer: Any, max_length: int) -> Dict:
    """Tokenize text input using the specified tokenizer."""
    embeddings = tokenizer.encode_plus(
        row["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.squeeze(0) for k, v in embeddings.items()}


def process_data(
    df: pd.DataFrame,
    tokenizer: Any,
    target_cols: List[str] = ["prompt"],
    max_length: int = 1536,
) -> Dataset:
    """Process DataFrame into a tokenized dataset."""
    dataset = Dataset.from_pandas(df[target_cols])
    return dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=max_length),
        batched=False,
        num_proc=1,
    )


@torch.no_grad()
@torch.amp.autocast("cuda")
def inference(
    df: pd.DataFrame, model: Any, target_tokens: List[int], batch_size: int, tokenizer: Any
) -> pd.DataFrame:
    """Perform model inference on the input data."""
    end_idx = 0
    logit_list = []

    for start_idx in tqdm(range(0, len(df)), total=len(df), desc="Inference"):
        if start_idx < end_idx:
            continue

        # Process batch
        end_idx = min(len(df), start_idx + batch_size)
        tmp = df.iloc[start_idx:end_idx].copy()
        dset = process_data(tmp, tokenizer)

        # Prepare inputs
        tmp["input_ids"] = dset["input_ids"]
        tmp["attention_mask"] = dset["attention_mask"]
        inputs = pad_without_fast_tokenizer_warning(
            tokenizer,
            {
                "input_ids": tmp["input_ids"].tolist(),
                "attention_mask": tmp["attention_mask"].tolist(),
            },
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
        ).to(model.device)

        # Get model outputs
        outputs = model(**inputs)
        logits = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

        # Extract last token logits
        last_token_logits = []
        for logit, mask in zip(logits, inputs["attention_mask"].cpu().numpy()):
            last_token_idx = mask.nonzero()[0][-1]
            last_token_logits.append(logit[last_token_idx, target_tokens])
        logit_list.extend(last_token_logits)

    df["logit"] = logit_list
    return df


def add_prompt(df: pd.DataFrame, mapping: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Create dataset with misconception options."""
    df["pred_ids"] = df["pred_ids"].apply(lambda x: list(map(int, x.split())))
    new_rows = []

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for i in range(params.topk // params.num_slide):
            if params.num_slide * i + params.num_choice > params.topk:
                break

            new_row = row.copy()
            mis_ids = row["pred_ids"][
                params.num_slide * i : params.num_slide * i + params.num_choice
            ]
            new_row["pred_ids"] = mis_ids

            # Format misconception names
            names = mapping.loc[mis_ids, "MisconceptionName"].tolist()
            names = "\n".join(
                [f"{x}: {y}" for x, y in zip(params.choice_words[: params.num_choice], names)]
            )

            new_row["mis_names"] = names
            new_row["idx"] = idx
            start_idx = params.num_slide * i
            end_idx = params.num_slide * i + params.num_choice
            new_row["rank"] = f"{start_idx + 1}-{end_idx}"
            new_rows.append(new_row)

    # Create final DataFrame
    df = pd.DataFrame(new_rows)
    df["last_choice"] = params.choice_words[-1]
    df["prompt"] = df.apply(
        lambda x: NA_PROMPT_FORMAT.format(**x) if params.add_na else PROMPT_FORMAT.format(**x),
        axis=1,
    )
    return df


def main(
    config: str = "/kaggle/working/reranker2.yaml",
) -> None:
    """Main function to run the inference pipeline."""
    # Load configuration
    cfg = OmegaConf.load(config)
    params = cfg.inference_listwise
    set_seed(params.seed)

    # Load data
    mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
    df = pd.read_parquet(Path(cfg.save_dir) / params.input_name)
    # Initialize tokenizer
    model_name = params.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    # Prepare choice tokens
    params.choice_words = get_choice_words(params.num_choice)
    params.choice_tokens = [tokenizer.encode(x)[0] for x in params.choice_words]

    # Process dataset
    tmp_df = add_prompt(df.copy(), mapping, params)
    tmp_df["length"] = tmp_df["prompt"].apply(lambda x: len(x.split()))
    tmp_df.sort_values("length", inplace=True, ascending=False)
    

    model = LLM(
        model=params.model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.99,
        max_logprobs=52,
        dtype="half",
        max_model_len=2048,
        enforce_eager=True,
        tensor_parallel_size=2
    )

    # Generate predictions
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=params.num_choice)
    outputs = model.generate(tmp_df["prompt"].tolist(), sampling_params)

    # Process outputs
    logits = []
    for output in outputs:
        output = output.outputs[0].logprobs[0]
        score_dict = {i: 0.0 for i in range(52)}
        for k in output.keys():
            if k in params.choice_tokens:
                score_dict[params.choice_tokens.index(k)] = math.exp(output[k].logprob)
        logits.append(list(score_dict.values()))

    tmp_df["logit"] = logits

    # Aggregate results
    new_rows = []
    for i, g in tmp_df.groupby("idx"):
        id_dict = defaultdict(list)
        for idx, row in g.iterrows():
            for pred_id, logit in zip(row["pred_ids"], row["logit"]):
                id_dict[pred_id].append(logit)
        id_dict = {k: np.mean(v) for k, v in id_dict.items()}
        sorted_ids = sorted(id_dict, key=id_dict.get, reverse=True)
        row = g.iloc[0].copy()
        row["pred_ids"] = sorted_ids
        new_rows.append(row)

    tmp_df = pd.DataFrame(new_rows)
    tmp_df["pred_ids"] = tmp_df["pred_ids"].apply(lambda x: " ".join(map(str, x)))
    tmp_df.to_parquet(Path(cfg.save_dir) / params.save_name, index=False)
if __name__ == "__main__":
    typer.run(main)



!python reranker2.py


# %%writefile reranker3.yaml
# save_dir: "/kaggle/working"
# input_dir: "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
# inference_listwise:
#   model_name: "/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1"
#   input_name: "data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_2.parquet"
#   save_name: "data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_3.parquet"
#   batch_size: 2
#   seed: 42
#   topk: 25
#   num_choice: 5
#   num_slide: 5

#   add_na: True
#   max_length: 2048
#   load_in_4bit: false


# %%writefile reranker3.py
# import math
# import string
# from collections import defaultdict
# from functools import partial
# from pathlib import Path
# from typing import Any, Dict, List

# import numpy as np
# import pandas as pd
# import torch
# import typer
# from datasets import Dataset
# from datasets.utils.logging import disable_progress_bar
# from omegaconf import OmegaConf
# from tqdm import tqdm
# from transformers import AutoTokenizer, set_seed
# from transformers.data.data_collator import pad_without_fast_tokenizer_warning
# from vllm import LLM, SamplingParams
# import torch

# # Disable progress bar for datasets
# disable_progress_bar()

# PROMPT_FORMAT: str = """<|im_start|>system
# You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
# Please return the most appropriate option from the list of misconceptions. Do not output anything other than options.<|im_end|>
# <|im_start|>user
# # Math Problem
# Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# # Misconception List
# {mis_names}<|im_end|>
# <|im_start|>assistant
# """

# NA_PROMPT_FORMAT: str = """<|im_start|>system
# You will be given math problem, overview of ther problem, correct answer, incorrect answer, and incorrect reason.
# Please return the most appropriate option from the list of misconceptions. Do not output anything other than options. If there are no suitable options, return NA.<|im_end|>
# <|im_start|>user
# # Math Problem
# Problem: {QuestionText}\nOverview: ({SubjectName}){ConstructName}\nCorrectAnswer: {Correct}\nIncorrectAnswer: {Answer}\nIncorrectReason: {kd}

# # Misconception List (rank: {rank})
# {mis_names}<|im_end|>
# <|im_start|>assistant
# """


# def get_choice_words(num_choices: int) -> List[str]:
#     """Generate a list of choice identifiers (A, B, C, etc.)."""
#     alphabets = list(string.ascii_uppercase + string.ascii_lowercase)
#     return alphabets[:num_choices]


# def tokenize_function(row: Dict, tokenizer: Any, max_length: int) -> Dict:
#     """Tokenize text input using the specified tokenizer."""
#     embeddings = tokenizer.encode_plus(
#         row["prompt"],
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_length,
#         add_special_tokens=False,
#     )
#     return {k: v.squeeze(0) for k, v in embeddings.items()}


# def process_data(
#     df: pd.DataFrame,
#     tokenizer: Any,
#     target_cols: List[str] = ["prompt"],
#     max_length: int = 1536,
# ) -> Dataset:
#     """Process DataFrame into a tokenized dataset."""
#     dataset = Dataset.from_pandas(df[target_cols])
#     return dataset.map(
#         partial(tokenize_function, tokenizer=tokenizer, max_length=max_length),
#         batched=False,
#         num_proc=1,
#     )


# @torch.no_grad()
# @torch.amp.autocast("cuda")
# def inference(
#     df: pd.DataFrame, model: Any, target_tokens: List[int], batch_size: int, tokenizer: Any
# ) -> pd.DataFrame:
#     """Perform model inference on the input data."""
#     end_idx = 0
#     logit_list = []

#     for start_idx in tqdm(range(0, len(df)), total=len(df), desc="Inference"):
#         if start_idx < end_idx:
#             continue

#         # Process batch
#         end_idx = min(len(df), start_idx + batch_size)
#         tmp = df.iloc[start_idx:end_idx].copy()
#         dset = process_data(tmp, tokenizer)

#         # Prepare inputs
#         tmp["input_ids"] = dset["input_ids"]
#         tmp["attention_mask"] = dset["attention_mask"]
#         inputs = pad_without_fast_tokenizer_warning(
#             tokenizer,
#             {
#                 "input_ids": tmp["input_ids"].tolist(),
#                 "attention_mask": tmp["attention_mask"].tolist(),
#             },
#             padding="longest",
#             pad_to_multiple_of=None,
#             return_tensors="pt",
#         ).to(model.device)

#         # Get model outputs
#         outputs = model(**inputs)
#         logits = torch.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

#         # Extract last token logits
#         last_token_logits = []
#         for logit, mask in zip(logits, inputs["attention_mask"].cpu().numpy()):
#             last_token_idx = mask.nonzero()[0][-1]
#             last_token_logits.append(logit[last_token_idx, target_tokens])
#         logit_list.extend(last_token_logits)

#     df["logit"] = logit_list
#     return df


# def add_prompt(df: pd.DataFrame, mapping: pd.DataFrame, params: Any) -> pd.DataFrame:
#     """Create dataset with misconception options."""
#     df["pred_ids"] = df["pred_ids"].apply(lambda x: list(map(int, x.split())))
#     new_rows = []

#     # Process each row
#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         for i in range(params.topk // params.num_slide):
#             if params.num_slide * i + params.num_choice > params.topk:
#                 break

#             new_row = row.copy()
#             mis_ids = row["pred_ids"][
#                 params.num_slide * i : params.num_slide * i + params.num_choice
#             ]
#             new_row["pred_ids"] = mis_ids

#             # Format misconception names
#             names = mapping.loc[mis_ids, "MisconceptionName"].tolist()
#             names = "\n".join(
#                 [f"{x}: {y}" for x, y in zip(params.choice_words[: params.num_choice], names)]
#             )

#             new_row["mis_names"] = names
#             new_row["idx"] = idx
#             start_idx = params.num_slide * i
#             end_idx = params.num_slide * i + params.num_choice
#             new_row["rank"] = f"{start_idx + 1}-{end_idx}"
#             new_rows.append(new_row)

#     # Create final DataFrame
#     df = pd.DataFrame(new_rows)
#     df["last_choice"] = params.choice_words[-1]
#     df["prompt"] = df.apply(
#         lambda x: NA_PROMPT_FORMAT.format(**x) if params.add_na else PROMPT_FORMAT.format(**x),
#         axis=1,
#     )
#     return df


# def main(
#     config: str = "/kaggle/working/reranker3.yaml",
# ) -> None:
#     """Main function to run the inference pipeline."""
#     # Load configuration
#     cfg = OmegaConf.load(config)
#     params = cfg.inference_listwise
#     set_seed(params.seed)

#     # Load data
#     mapping = pd.read_csv(Path(cfg.input_dir) / "misconception_mapping.csv")
#     df = pd.read_parquet(Path(cfg.save_dir) / params.input_name)
#     # Initialize tokenizer
#     model_name = params.model_name
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.truncation_side = "left"
#     # Prepare choice tokens
#     params.choice_words = get_choice_words(params.num_choice)
#     params.choice_tokens = [tokenizer.encode(x)[0] for x in params.choice_words]

#     # Process dataset
#     tmp_df = add_prompt(df.copy(), mapping, params)
#     tmp_df["length"] = tmp_df["prompt"].apply(lambda x: len(x.split()))
#     tmp_df.sort_values("length", inplace=True, ascending=False)
    

#     model = LLM(
#         model=params.model_name,
#         trust_remote_code=True,
#         gpu_memory_utilization=0.99,
#         max_logprobs=52,
#         dtype="half",
#         max_model_len=2048,
#         enforce_eager=True,
#         tensor_parallel_size=2
#     )

#     # Generate predictions
#     sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=params.num_choice)
#     outputs = model.generate(tmp_df["prompt"].tolist(), sampling_params)

#     # Process outputs
#     logits = []
#     for output in outputs:
#         output = output.outputs[0].logprobs[0]
#         score_dict = {i: 0.0 for i in range(52)}
#         for k in output.keys():
#             if k in params.choice_tokens:
#                 score_dict[params.choice_tokens.index(k)] = math.exp(output[k].logprob)
#         logits.append(list(score_dict.values()))

#     tmp_df["logit"] = logits

#     # Aggregate results
#     new_rows = []
#     for i, g in tmp_df.groupby("idx"):
#         id_dict = defaultdict(list)
#         for idx, row in g.iterrows():
#             for pred_id, logit in zip(row["pred_ids"], row["logit"]):
#                 id_dict[pred_id].append(logit)
#         id_dict = {k: np.mean(v) for k, v in id_dict.items()}
#         sorted_ids = sorted(id_dict, key=id_dict.get, reverse=True)
#         row = g.iloc[0].copy()
#         row["pred_ids"] = sorted_ids
#         new_rows.append(row)

#     tmp_df = pd.DataFrame(new_rows)
#     tmp_df["pred_ids"] = tmp_df["pred_ids"].apply(lambda x: " ".join(map(str, x)))
#     tmp_df.to_parquet(Path(cfg.save_dir) / params.save_name, index=False)
# if __name__ == "__main__":
#     typer.run(main)


# !python reranker3.py


import pandas as pd

df_stage_2 = pd.read_parquet("/kaggle/working/data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_2.parquet")

# Chỉ giữ lại các cột cần thiết và merge theo QuestionId_Answer
merged = df_stage_2[['QuestionId_Answer', 'pred_ids']].rename(columns={'pred_ids': 'pred_stage_2'})



# Chỉ giữ lại các cột cần thiết và merge theo QuestionId_Answer
def split_id(row):
    stage2_ids = row['pred_stage_2'].split()[:25]
    return ' '.join(stage2_ids)
merged['pred_ids'] = merged.apply(split_id, axis=1)


# # Merge theo QuestionId_Answer
# merged = pd.merge(df_stage_3, df_stage_2, on='QuestionId_Answer', how='inner')

# def combine_predictions(row):
#     stage3_ids = row['pred_stage_3'].split()[:10]
#     stage2_ids = row['pred_stage_2'].split()[10:25]
#     return ' '.join(stage3_ids + stage2_ids)

# # Tạo cột mới với kết quả mong muốn
# merged['pred_ids'] = merged.apply(combine_predictions, axis=1)


submission = merged[['QuestionId_Answer','pred_ids']]
submission.rename(columns={"pred_ids": "MisconceptionId"}).to_csv("submission.csv", index=False)


pd.read_csv('submission.csv')


# !pip install gradio


# import pandas as pd
# import gradio as gr

# # Đọc dữ liệu
# df = pd.read_parquet('/kaggle/working/data_listwise_with_Qwen2.5_Math_7B_Instruct_stage_3.parquet').rename(columns={"pred_ids": "MisconceptionId"})  # chứa QuestionId_Answer và MisconceptionId
# mapping = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')  # chứa: MisconceptionId, MisconceptionName

# # Chuẩn hóa và merge
# df['MisconceptionId'] = df['MisconceptionId'].astype(str).str.split()
# df = df.explode('MisconceptionId')
# df['MisconceptionId'] = df['MisconceptionId'].astype(int)

# merged = df.merge(mapping, on='MisconceptionId', how='left')
# grouped = merged.groupby('QuestionId_Answer')
# # Hàm hiển thị
# def show_info(qid_answer):
#     group = grouped.get_group(qid_answer)
#     row = group.iloc[0]

#     question = row['QuestionText']
#     answer = row['Answer']
#     correct = row['Correct']

#     # Chỉ lấy 5 misconception đầu tiên
#     misconceptions = group['MisconceptionName'].dropna().unique().tolist()[:5]
#     misconceptions_html = "<ul>" + "".join(f"<li>{m}</li>" for m in misconceptions) + "</ul>"

#     html = f"""
#     <div id="mathjax-container">
#         <h3>Question:</h3>
#         <p>{question}</p>
#         <p><b>Answer:</b> {answer}</p>
#         <p><b>Correct:</b> {correct}</p>
#         <h4>Misconceptions (Top 5):</h4>
#         {misconceptions_html}
#     </div>

#     <!-- MathJax Script -->
#     <script>
#     window.MathJax = {{
#       tex: {{ inlineMath: [['\\(', '\\)']], displayMath: [['\\[', '\\]']] }},
#       svg: {{ fontCache: 'global' }}
#     }};
#     if (!window.mathjaxScriptLoaded) {{
#         const script = document.createElement('script');
#         script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
#         script.async = true;
#         script.onload = () => MathJax.typeset();
#         document.head.appendChild(script);
#         window.mathjaxScriptLoaded = true;
#     }} else {{
#         MathJax.typesetPromise();
#     }}
#     </script>
#     """
#     return html

# # UI
# dropdown = gr.Dropdown(choices=sorted(df['QuestionId_Answer'].unique()), label="Select QuestionId_Answer")

# iface = gr.Interface(
#     fn=show_info,
#     inputs=dropdown,
#     outputs=gr.HTML(),
#     title="Math Viewer with LaTeX",
#     theme="default"  # Sáng
# )

# iface.launch()




