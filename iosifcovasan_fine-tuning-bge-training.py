EXP_NAME = "fine-tuning-bge"
DATA_PATH = "/kaggle/input/eedi-mining-misconceptions-in-mathematics"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
COMPETITION_NAME = "eedi-mining-misconceptions-in-mathematics"
OUTPUT_PATH = "."
MODEL_OUTPUT_PATH = f"{OUTPUT_PATH}/trained_model"

RETRIEVE_NUM = 25

EPOCH = 2
LR = 2e-05
BS = 8
GRAD_ACC_STEP = 128 // BS

TRAINING = True
DEBUG = False
WANDB = False


%pip install -qq polars==1.7.1
%pip install -qq datasets==3.0.0
%pip install -qq sentence_transformers==3.1.0


import os
import numpy as np

from datasets import load_dataset, Dataset

import wandb
import polars as pl

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator


import datasets
import sentence_transformers

assert pl.__version__ == "1.7.1"
assert datasets.__version__ == "3.0.0"
assert sentence_transformers.__version__ == "3.1.0"


NUM_PROC = os.cpu_count()


if WANDB:
    # Settings -> add wandb api
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    wandb.login(key=user_secrets.get_secret("wandb_api"))
    wandb.init(project=COMPETITION_NAME, name=EXP_NAME)
    REPORT_TO = "wandb"
else:
    REPORT_TO = "none"

REPORT_TO


train = pl.read_csv(f"{DATA_PATH}/train.csv")
misconception_mapping = pl.read_csv(f"{DATA_PATH}/misconception_mapping.csv")


common_col = [
    "QuestionId",
    "ConstructName",
    "SubjectName",
    "QuestionText",
    "CorrectAnswer",
]

train_long = (
    train
    .select(
        pl.col(common_col + [f"Answer{alpha}Text" for alpha in ["A", "B", "C", "D"]])
    )
    .unpivot(
        index=common_col,
        variable_name="AnswerType",
        value_name="AnswerText",
    )
    .with_columns(
        pl.concat_str(
            [
                pl.col("ConstructName"),
                pl.col("SubjectName"),
                pl.col("QuestionText"),
                pl.col("AnswerText"),
            ],
            separator=" ",
        ).alias("AllText"),
        pl.col("AnswerType").str.extract(r"Answer([A-D])Text$").alias("AnswerAlphabet"),
    )
    .with_columns(
        pl.concat_str(
            [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
        ).alias("QuestionId_Answer"),
    )
    .sort("QuestionId_Answer")
)
train_long.head()


train_misconception_long = (
    train.select(
        pl.col(
            common_col + [f"Misconception{alpha}Id" for alpha in ["A", "B", "C", "D"]]
        )
    )
    .unpivot(
        index=common_col,
        variable_name="MisconceptionType",
        value_name="MisconceptionId",
    )
    .with_columns(
        pl.col("MisconceptionType")
        .str.extract(r"Misconception([A-D])Id$")
        .alias("AnswerAlphabet"),
    )
    .with_columns(
        pl.concat_str(
            [pl.col("QuestionId"), pl.col("AnswerAlphabet")], separator="_"
        ).alias("QuestionId_Answer"),
    )
    .sort("QuestionId_Answer")
    .select(pl.col(["QuestionId_Answer", "MisconceptionId"]))
    .with_columns(pl.col("MisconceptionId").cast(pl.Int64))
)

train_misconception_long.head()


# join MisconceptionId
train_long = train_long.join(train_misconception_long, on="QuestionId_Answer")
train_long.head()


model = SentenceTransformer(MODEL_NAME)

train_long_vec = model.encode(
    train_long["AllText"].to_list(), normalize_embeddings=True
)
misconception_mapping_vec = model.encode(
    misconception_mapping["MisconceptionName"].to_list(), normalize_embeddings=True
)
print(train_long_vec.shape)
print(misconception_mapping_vec.shape)


train_cos_sim_arr = cosine_similarity(train_long_vec, misconception_mapping_vec)
train_sorted_indices = np.argsort(-train_cos_sim_arr, axis=1)


train_long = train_long.with_columns(
    pl.Series(train_sorted_indices[:, :RETRIEVE_NUM].tolist()).alias(
        "PredictMisconceptionId"
    )
)


train_retrieved = (
    train_long.filter(
        pl.col(
            "MisconceptionId"
        ).is_not_null()  # TODO: Consider ways to utilize data where MisconceptionId is NaN.
    )
    .explode("PredictMisconceptionId")
    .join(
        misconception_mapping,
        on="MisconceptionId",
    )
    .join(
        misconception_mapping.rename(lambda x: "Predict" + x),
        on="PredictMisconceptionId",
    )
)
train_retrieved.shape


train = (
    Dataset.from_polars(train_retrieved)
    .filter(  # To create an anchor, positive, and negative structure, delete rows where the positive and negative are identical.
        lambda example: example["MisconceptionId"] != example["PredictMisconceptionId"],
        num_proc=NUM_PROC,
    )
)


train


if DEBUG:
    train = train.select(range(1000))
    EPOCH = 1


model = SentenceTransformer(MODEL_NAME)

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=OUTPUT_PATH,
    # Optional training parameters:
    num_train_epochs=EPOCH,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GRAD_ACC_STEP,
    per_device_eval_batch_size=BS,
    eval_accumulation_steps=GRAD_ACC_STEP,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    lr_scheduler_type="cosine_with_restarts",
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=2,
    logging_steps=100,
    report_to=REPORT_TO,  # Will be used in W&B if `wandb` is installed
    run_name=EXP_NAME,
    do_eval=False
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train.select_columns(
        ["AllText", "MisconceptionName", "PredictMisconceptionName"]
    ),
    loss=loss
)

trainer.train()
model.save_pretrained(MODEL_OUTPUT_PATH)

