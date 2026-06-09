!pip install -U scikit-learn
!pip install -U bitsandbytes


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from typing import Set
from sklearn.model_selection import GroupKFold
import os

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Set

import typer
from datasets import Dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives
from transformers import set_seed
from transformers import EarlyStoppingCallback

from transformers import  BitsAndBytesConfig



%%writefile exp_gpu.yaml

input_dir: "input"
save_dir: "exp_output"
best_model_dir: "best_models"

# generate_question.py
generate_question:
  model_name: "google/gemini-pro-1.5"
  save_name: "additonal_question.csv"
  embedding_model: "dunzhang/stella_en_1.5B_v5"
  num_shot: 4
  sem_number: 100 # number of conccurent

# split_fold.py
split_fold:
  n_split: 5
  add_data_name: "additonal_question.csv"
  seed: 42
  save_name: "data.csv"

# distill.py
distill:
  model_name: "Qwen/Qwen2.5-32B-Instruct-AWQ"
  input_name: "data.csv"
  save_name: "data_kd.csv"

# train_biencoder.py
train_biencoder:
  model_name: "dunzhang/stella_en_1.5B_v5"
  input_name: "data.csv"
  output_dir: "output_bi_1.5B"
  is_lora: true
  load_in_4bit: False
  mini_batch_size: 1
  seed: 42
  lora_config:
    r: 32
    lora_alpha: 64
  hard_negative_params:
    range_min: 512
    num_negatives: 2
    batch_size: 32
  train_args:
    num_train_epochs: 1.0
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 2
    learning_rate: 0.0005
    warmup_steps: 0
    eval_strategy: epoch
    save_only_model: true
    # eval_steps: 1
    metric_for_best_model: val_cosine_recall@100
    load_best_model_at_end: true
    greater_is_better: true
    save_strategy: epoch
    # save_steps: 1
    lr_scheduler_type: "cosine"
    save_total_limit: 1
    logging_steps: 1
    report_to: "none"
    bf16: true





# inference_biencoder.py
inference_biencoder:
  model_output_dir: "output_bi_1.5B"
  input_name: "data_kd.csv"
  save_name: "data_bi.csv"
  model_name: "dunzhang/stella_en_1.5B_v5"
  is_lora: true
  load_in_4bit: false
  batch_size: 32

# train_listwise.py
train_listwise:
  model_name: "unsloth/Qwen2.5-32B-Instruct"
  input_name: "data_bi.csv"
  output_dir: "output_listwise"
  add_na: true # add NA to the options
  num_choice: 52
  num_slide: 52
  train_negative_topk: 208 # this is only used when add_na is true
  train_topk: 208
  inference_topk: 52
  max_length: 1900
  seed: 42
  load_in_4bit: false
  lora_config:
    r: 24
    lora_alpha: 48
  train_args:
    per_device_train_batch_size: 4
    per_device_eval_batch_size: 1
    gradient_accumulation_steps: 2
    num_train_epochs: 1.0
    learning_rate: 5e-5
    warmup_steps: 10
    logging_steps: 10
    overwrite_output_dir: true
    save_total_limit: 2
    lr_scheduler_type: "cosine"
    report_to: "none"
    bf16: true
    eval_strategy: "steps"
    metric_for_best_model: "loss"

# inference_listwise.py
inference_listwise:
  model_output_dir: "output_listwise"
  input_name: "data_bi.csv"
  save_name: "data_listwise.csv"
  batch_size: 2
  seed: 42
  topk: 104
  num_choice: 52
  num_slide: 52
  add_na: True
  max_length: 2048
  load_in_4bit: false



config = "exp_gpu.yaml"


# cfg = OmegaConf.load(config)
# params = cfg.train_biencoder
# params.train_args


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame to create a structured dataset for misconception analysis.

    Args:
        df: Input DataFrame containing answer and misconception data

    Returns:
        Processed DataFrame with restructured data
    """
    datas: List[Dict[str, Union[str, int]]] = []
    for col in ["A", "B", "C", "D"]:
        answer_col = f"Answer{col}Text"
        misconception_col = f"Misconception{col}Id"

        for _, row in df.iterrows():
            if pd.isna(row[misconception_col]):
                continue

            correct_col = row["CorrectAnswer"]
            correct_answer_col = f"Answer{correct_col}Text"

            data = {
                "Answer": row[answer_col],
                "Correct": row[correct_answer_col],
                "MisconceptionId": int(row[misconception_col]),
            }

            if correct_col == col:
                continue

            row_dict = row.to_dict()
            # Remove unnecessary columns
            for c in ["A", "B", "C", "D"]:
                row_dict.pop(f"Misconception{c}Id")
                row_dict.pop(f"Answer{c}Text")

            data.update(row_dict)
            datas.append(data)

    return pd.DataFrame(datas)


mis_competition = pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv')
train_competition =  pd.read_csv('/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv')

print(train_competition.shape)
train_systhetic = pd.read_csv('/kaggle/input/eedi-mcq-dataset/train.csv')
# print(train_systhetic.shape)

misconception_columns = ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']
misconception_list = mis_competition['MisconceptionId']

# 创建一个布尔掩码，检查每一行的所有Misconception列是否都在列表中或为NaN
mask = train_systhetic[misconception_columns].apply(
    lambda x: x.isin(misconception_list) | x.isna()
).all(axis=1)
train_systhetic_filtered = train_systhetic[mask]
train_systhetic_filtered = train_systhetic_filtered[~train_systhetic_filtered['QuestionText'].isin(train_competition['QuestionText'])]
# train_systhetic_filtered = train_systhetic_filtered.sample(2000,random_state=42)
print(train_systhetic_filtered.shape)


val_mask = ~train_systhetic_filtered['SubjectName'].isin(train_competition['SubjectName'])
val = train_systhetic_filtered[val_mask]
print(val.shape)

val = val.sample(n=2800,random_state=42)
val_df_1 = val.iloc[:2000].copy(True)
val_df = val.iloc[2000:].copy(True)
train_df = pd.concat([train_competition,val_df_1],axis=0,ignore_index=True)
train_df.shape,val_df.shape


INPUT_DIR_1 = "/kaggle/input/eedi-1-misconception-explanation"
df_map = pd.read_parquet(f"{INPUT_DIR_1}/misconception_mapping.parquet")
df_map["Misconception_concat"] = df_map["MisconceptionName"] + "\n" + df_map["Misconception_explain_1"]
df_map=df_map[['MisconceptionName','Misconception_concat']]


mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
mapping = mapping.merge(df_map, on="MisconceptionName")

mapping = mapping.drop('MisconceptionName', axis=1)
mapping = mapping.rename(columns={"Misconception_concat": "MisconceptionName"})


mapping.head()



# mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

train_df = load_data(train_df).reset_index(drop=True)
train_df["original"] = True
train_df = train_df.merge(mapping, on="MisconceptionId")

val_df = load_data(val_df).reset_index(drop=True)
val_df["original"] = True
val_df = val_df.merge(mapping, on="MisconceptionId")
print(train_df.shape,val_df.shape)



train_df.head()






def calculate_misconception_overlap(train_misid: Set[int], val_misid: Set[int]) -> float:
    """Calculate the overlap ratio between training and validation misconceptions.

    Args:
        train_misid: Set of misconception IDs in training set
        val_misid: Set of misconception IDs in validation set

    Returns:
        float: Overlap ratio (1 - intersection/validation size)
    """
    return 1.0 - len(train_misid & val_misid) / len(val_misid)


def split_dataset(df: pd.DataFrame, n_splits: int, group_col: str) -> pd.DataFrame:
    """Split dataset into folds using GroupKFold.

    Args:
        df: Input DataFrame
        n_splits: Number of splits
        group_col: Column name to use for grouping

    Returns:
        pd.DataFrame: DataFrame with added fold column
    """
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
        if group_col == "QuestionId":
            # Calculate overlap for original data
            train_misid = set(df.loc[train_idx]["MisconceptionId"])
            val_misid = set(df.loc[val_idx]["MisconceptionId"])
            overlap = calculate_misconception_overlap(train_misid, val_misid)
            print(f"Fold {fold_idx} misconception overlap: {overlap:.3f}")

        df.loc[val_idx, "fold"] = fold_idx

    return df


# cfg = OmegaConf.load(config)
# params = cfg.split_fold
# set_seed(params.seed)

# # Load misconception mapping
# mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

# # Process original training data
# df = load_data(train_competition).reset_index(drop=True)
# df["original"] = True
# df = split_dataset(df, params.n_split, "QuestionId")

# # Process synthetic data
# synthetic_df =  load_data(train_systhetic_filtered).reset_index(drop=True)
# synthetic_df["original"] = False
# synthetic_df = split_dataset(synthetic_df, params.n_split, "SubjectName")

# # Combine datasets
# combined_df = pd.concat([df, synthetic_df], ignore_index=True)
# combined_df["fold"] = combined_df["fold"].astype(int)

# # Validate number of folds
# if combined_df["fold"].nunique() != params.n_split:
#     raise ValueError(f"Expected {params.n_split} folds, got {combined_df['fold'].nunique()}")

# # Merge with misconception mapping and save
# final_df = combined_df.merge(mapping, on="MisconceptionId")
# print(final_df.shape)
# output_path = Path(cfg.save_dir) / params.save_name
# if not os.path.exists(Path(cfg.save_dir)):
#     os.makedirs(Path(cfg.save_dir))
# final_df.to_csv(output_path, index=False)
# print(f"Saved split dataset to {output_path}")

# final_df.head()



# %%writefile train_retriever.py

PROMPT_FORMAT: str = """Subject: {SubjectName}
Construct: {ConstructName}
Question: {QuestionText}
CorrectAnswer: {Correct}
IncorrectAnswer: {Answer}
"""


def create_val(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Create validation dataset by merging dataframe with mapping and adding labels.

    Args:
        df: Input DataFrame containing the base data
        mapping: DataFrame containing misconception mapping information

    Returns:
        DataFrame with processed validation data
    """
    df = df.merge(mapping, how="cross")
    df["label"] = 0
    df.loc[df["MisconceptionId_x"] == df["MisconceptionId_y"], "label"] = 1
    target_cols = ["prompt", "MisconceptionName_y", "label"]
    df = df[target_cols].rename(columns={"MisconceptionName_y": "MisconceptionName"})
    return df


def create_evaluator(df: pd.DataFrame, name: str = "train") -> InformationRetrievalEvaluator:
    """
    Create an evaluator for information retrieval tasks.

    Args:
        df: DataFrame containing prompts, misconception names, and labels
        name: Name identifier for the evaluator

    Returns:
        Configured InformationRetrievalEvaluator object
    """
    relevant_docs: DefaultDict[str, Set[str]] = defaultdict(set)
    queries: Dict[str, str] = {str(k): v for k, v in enumerate(df["prompt"].unique())}
    corpus: Dict[str, str] = {str(k): v for k, v in enumerate(df["MisconceptionName"].unique())}

    # Create reverse mappings for efficient lookup
    qid_dict: Dict[str, str] = {v: k for k, v in queries.items()}
    cid_dict: Dict[str, str] = {v: k for k, v in corpus.items()}

    # Build relevant documents mapping
    for prompt, g in df.groupby("prompt"):
        for mis_name, label in g[["MisconceptionName", "label"]].values:
            if label == 1:
                qid = qid_dict[str(prompt)]
                cid = cid_dict[mis_name]
                relevant_docs[qid].add(cid)

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        map_at_k=[25],
        mrr_at_k=[25],
        precision_recall_at_k=[50, 100, 150, 200],
        ndcg_at_k=[25],
        accuracy_at_k=[25],
    )


def train_retriever(train_df,val_df,config):
    cfg = OmegaConf.load(config)
    params = cfg.train_biencoder
    set_seed(params.seed)
    
    # Load and prepare data
    # df = pd.read_csv(Path(cfg.save_dir) / params.input_name)
    # df = final_df
    # mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
    train_df["prompt"] = train_df.apply(lambda x: PROMPT_FORMAT.format(**x), axis=1)
    val_df["prompt"] = val_df.apply(lambda x: PROMPT_FORMAT.format(**x), axis=1)

    # Split data into train and validation sets
    # train_df = df.loc[df.fold != fold].copy()
    # val_df = df.loc[(df.fold == fold) & (df.original)].copy()
    val_df = create_val(val_df, mapping)

    # Create dataset for training
    train_dset = Dataset.from_dict(
        {
            "anchor": train_df["prompt"].tolist(),
            "positive": train_df["MisconceptionName"].tolist(),
        }
    )

    # Setup paths and wandb
    name = f"fold_val"
    output_dir = str(Path(cfg.save_dir) / params.output_dir / name)
    best_model_path = str(Path(cfg.best_model_dir) / params.output_dir / name)
    # wandb.init(project="eedi-biencoder", name=f"{name}_{params.model_name.split('/')[-1]}")

    # Initialize model
    model = SentenceTransformer(
        params.model_name,
        trust_remote_code=True,
        model_kwargs={"load_in_4bit": params.load_in_4bit},
    )

    # Perform hard negative mining
    train_dset = mine_hard_negatives(
        train_dset,
        model,
        **params["hard_negative_params"],
    )

    # Add LoRA adapter if specified
    if params.is_lora:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            target_modules="all-linear",
            lora_dropout=0.01,
            **params["lora_config"],
        )
        model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
        model[0].auto_model.print_trainable_parameters()

    # Setup loss function and evaluator
    loss = losses.CachedMultipleNegativesRankingLoss(
        model, mini_batch_size=params.mini_batch_size, show_progress_bar=True
    )
    val_evaluator = create_evaluator(val_df, name="val")

    # Configure training arguments
    args = SentenceTransformerTrainingArguments(
        **params.train_args,
        output_dir=output_dir,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )

    early_stopper = EarlyStoppingCallback(
        early_stopping_patience=10, # you can change this value if needed
        early_stopping_threshold=0.01 # you can change this value if needed
    )
    
    # Initialize and run trainer
    trainer = SentenceTransformerTrainer(
        args=args,
        model=model,
        train_dataset=train_dset,
        loss=loss,
        evaluator=val_evaluator,
        callbacks=[early_stopper],

    )
    trainer.train()
    trainer.save_model(best_model_path)



train_retriever(train_df,val_df,config)













