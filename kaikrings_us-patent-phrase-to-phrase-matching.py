!pip install git+https://github.com/kkrings/fastts


import numpy as np
import pandas as pd
from fastai.text.all import ColReader, DataBlock, Learner, PearsonCorrCoef, RandomSplitter, RegressionBlock
from fastts.callbacks.classification.sequence import sequence_classification
from fastts.transforms.block import TransformersTextBlock
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")


df.head()


def add_input(df):
    df["input"] = ("TEXT1: " + df.context + "; TEXT2: " + df.target + "; ANC1: " + df.anchor).astype("string")


add_input(df)


df.head()


config = AutoConfig.from_pretrained("microsoft/deberta-v3-small")


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False)
tokenizer.model_max_length = config.max_position_embeddings


dblock = DataBlock(
    blocks=(TransformersTextBlock(tokenizer, truncation=True), RegressionBlock),
    get_x=ColReader("input"),
    get_y=ColReader("score"),
    splitter=RandomSplitter(0.25),
)


dls = dblock.dataloaders(df, bs=128)


dls.show_batch(max_n=5)


model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small", num_labels=1)


learn = Learner(
    dls,
    model,
    cbs=list(sequence_classification(loss_from_model=True)),
    metrics=[PearsonCorrCoef()],
).to_fp16()


# learn.lr_find()


learn.fine_tune(4, 8e-5)


learn.show_results()




