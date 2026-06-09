import os
import re
import sys
import datetime
import requests
import itertools
import functools
import logging
import shutil
import json
import warnings
import joblib
import gc
import random
import string
import re
import collections

import nltk
import pandas as pd
import numpy as np

import pytorch_lightning as pl

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from scipy.special import softmax
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau


class Conf_cl:
    
    target_data_name = "Mod-deberta-v3-large-jigsaw-tox"
    inference_launch_only = True
    use_pretrain_model_param = False
    target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    model_name = "microsoft/deberta-v3-large"
    hid_size = 1024
    head = 256
    tail = 0
    max_length = head + tail

    n_fold = 5
    train_fold_num = [0, 1, 2, 3, 4]
    gradient_clip_val = 100
    accumulate_grad_batches = 2
    max_epochs = 4
    early_stopping = False

    train_batch_size = 4
    valid_batch_size = 8
    num_workers = 4
    resume_from_checkpoint = None
    
    optimizer = dict(
        optimizer="AdamW", 
        lr=1e-5, 
        weight_decay=1e-5)
    
    scheduler = dict(
        interval = "step",
        scheduler="get_cosine_schedule_with_warmup",
        num_warmup_steps=200,
        num_cycles=0.5)

    seed_num = 2025
    
    kaggle_dataset_path = "../input/mod-deberta-v3-large-jigsaw-tox"
    


# Util functions

def seed_fun(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_fun(cfg):

    cfg.input = f"../input"

    cfg.input_jigsaw_01 = os.path.join(cfg.input, "jigsaw-toxic-comment-classification-challenge")
    cfg.input_jigsaw_02 = os.path.join(cfg.input, "jigsaw-unintended-bias-in-toxicity-classification")
    cfg.input_jigsaw_03 = os.path.join(cfg.input, "jigsaw-toxic-severity-rating")
    cfg.input_ruddit = os.path.join(cfg.input, "ruddit-jigsaw-dataset")
    cfg.jigsaw_inputs = [cfg.input_jigsaw_01, 
                         cfg.input_jigsaw_02, 
                         cfg.input_jigsaw_03, 
                         cfg.input_ruddit]

    cfg.exper = "./"

    if cfg.kaggle_dataset_path is not None:
        cfg.exper_model = os.path.join(cfg.kaggle_dataset_path, "model")
    else:
        cfg.exper_model = os.path.join(cfg.exper, "model")

    cfg.submission = "./"
    
    cfg.exper_pred = os.path.join(cfg.exper, "preds")

    make_dirs = [cfg.exper_model, cfg.exper_pred]
    
    for d in make_dirs:
        os.makedirs(d, exist_ok=True)
            
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    warnings.filterwarnings("ignore")
    seed_fun(cfg.seed_num)

    return cfg



# Setup data
Conf_obj = setup_fun(Conf_cl)


# Dataset

class CompetTrainDataset(Dataset):
    def __init__(self, cfg, df, tokenizer, text_col):
        self.cfg = cfg
        self.comment_text = df[text_col].values
        self.targets = df[cfg.target_cols].values
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, idx):

        text = str(self.comment_text[idx])
        inputs = prepare_input(self.cfg, text, self.tokenizer)
        targets = torch.tensor(self.targets[idx]).float()

        return inputs, targets


class CompetTestDataset(Dataset):
    def __init__(self, cfg, df, tokenizer, text_col):
        self.cfg = cfg
        self.comment_text = df[text_col].fillna("none").values
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, idx):
        text = str(self.comment_text[idx])
        inputs = prepare_input(self.cfg, text, self.tokenizer)
        return inputs


def prepare_input(cfg, text, tokenizer):
    if cfg.tail == 0:
        inputs = tokenizer.encode_plus(
            text, 
            return_tensors=None, 
            add_special_tokens=True, 
            max_length=cfg.max_length,
            pad_to_max_length=True,
            truncation=True)
        
        for ks, vs in inputs.items():
            inputs[ks] = torch.tensor(vs, dtype=torch.long)

    else:
        inputs = tokenizer.encode_plus(
            text,
            return_tensors=None, 
            add_special_tokens=True, 
            truncation=True)
        
        for ks, vs in inputs.items():
            vs_length = len(vs)
            if vs_length > cfg.max_length:
                vs = np.hstack([vs[:cfg.head], vs[-cfg.tail:]])

            if ks == 'input_ids':
                new_vs = np.ones(cfg.max_length) * tokenizer.pad_token_id

            else:
                new_vs = np.zeros(cfg.max_length)

            new_vs[:vs_length] = vs 
            inputs[ks] = torch.tensor(new_vs, dtype=torch.long)

    return inputs


class CompetDataModule(pl.LightningDataModule):
    def __init__(self, cfg, tokenizer, train_df, valid_df, text_col):
        super(CompetDataModule).__init__()

        self.cfg = cfg
        self.text_col = text_col
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.valid_df = valid_df

        self.train_dataset = None
        self.val_dataset = None

        self.allow_zero_length_dataloader_with_multiple_devices = False

    def setup(self, stage=None):
        self.train_dataset = CompetTrainDataset(
            cfg=self.cfg, df=self.train_df, tokenizer=self.tokenizer, text_col=self.text_col)
        self.val_dataset = CompetTrainDataset(
            cfg=self.cfg, df=self.valid_df, tokenizer=self.tokenizer, text_col=self.text_col)
        
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=self.cfg.train_batch_size, 
            shuffle=True, 
            num_workers=self.cfg.num_workers, 
            pin_memory=True, 
            drop_last=True)
        
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.valid_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers, 
            pin_memory=True, 
            drop_last=False)

        return val_dataloader

    def _log_hyperparams(self, params):  # заглушка
        pass


# Model

def optimizer_fun(cfg, parameters):
    opt = cfg.optimizer
    if opt["optimizer"] == "AdamW":
        optimizer = AdamW(
            parameters,
            lr=opt["lr"],
            weight_decay=opt["weight_decay"]
            )
    
    elif opt["optimizer"] == "Adam":
        optimizer = Adam(
            parameters,
            lr=opt["lr"],
            weight_decay=opt["weight_decay"]
            )
    
    else:
        raise NotImplementedError
    
    return optimizer


def scheduler_fun(cfg, optimizer, num_train_steps):
    sch = cfg.scheduler
    if sch["scheduler"] == "get_linear_schedule_with_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=sch["num_warmup_steps"],
            num_training_steps=num_train_steps)
    
    elif sch["scheduler"] == "get_cosine_schedule_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=sch["num_warmup_steps"],
            num_training_steps=num_train_steps,
            num_cycles=sch["num_cycles"]
            )

    elif sch["scheduler"] == "MultiStepLR":
        scheduler = MultiStepLR(
            optimizer, 
            milestones=sch["milestones"], 
            gamma=sch["gamma"]
        )

    else:
        raise NotImplementedError
    
    return scheduler


class CompetModel(pl.LightningModule):
    def __init__(self, cfg):
        super(CompetModel, self).__init__()
        self.cfg = cfg
        self.total_steps = None
        self.dataset_size = None

        self.backborn = backborn_fun(cfg)   
        self.out = nn.Linear(cfg.hid_size, len(cfg.target_cols))

    def forward(self, inputs):
        x = self.backborn(**inputs)
        x = x[0]
        x = x[:, 0, :]
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        self.log("val_loss", loss, on_step=True, logger=True, prog_bar=True)
        return loss

    def loss(self, outputs, targets):
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, targets)
        loss = torch.sqrt(loss)
        return loss

    def setup(self, stage=None):
        if stage != "fit":
            return

        if self.dataset_size is None:
            dataset = self.trainer._data_connector._train_dataloader_source.dataloader()
            self.dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.cfg.train_batch_size * self.trainer.accumulate_grad_batches * num_devices
        self.total_steps = (self.dataset_size // effective_batch_size) * self.cfg.max_epochs

    def configure_optimizers(self):
        optimizer = optimizer_fun(self.cfg, parameters=self.parameters())

        if self.cfg.scheduler is None:
            return [optimizer]
        else:
            scheduler = scheduler_fun(self.cfg, optimizer, num_train_steps=self.total_steps)
            return [optimizer], [{"scheduler": scheduler, "interval": self.cfg.scheduler["interval"]}]



# Load model

def tokenizer_fun(cfg):

    pretrained_dir = os.path.join(cfg.exper_model, "pretrain_param")
    
    tokenizer_path = os.path.join(pretrained_dir, "tokenizer_config.json")

    if not os.path.isfile(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        tokenizer.save_pretrained(pretrained_dir)
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

    return tokenizer


def backborn_fun(cfg):
    
    pretrained_dir = os.path.join(cfg.exper_model, "pretrain_param")
    
    backborn_path = os.path.join(pretrained_dir, "pytorch_model.bin")

    if not os.path.isfile(backborn_path):
        
        model_config = AutoConfig.from_pretrained(cfg.model_name)

        model_config.attention_probs_dropout_prob = 0.0 # no dropout
        model_config.hidden_dropout_prob = 0.0 # no dropout

        backborn = AutoModel.from_pretrained(cfg.model_name, config=model_config)

        backborn.save_pretrained(pretrained_dir)
    
    else:
        
        model_config = AutoConfig.from_pretrained(pretrained_dir)

        model_config.attention_probs_dropout_prob = 0.0 # no dropout
        model_config.hidden_dropout_prob = 0.0 # no dropout
        
        if cfg.use_pretrain_model_param:
            backborn = AutoModel.from_pretrained(pretrained_dir, config=model_config)
        else:
            backborn = AutoModel.from_config(model_config)

    return backborn


# Metrics

def validation_data_hat_fun(cfg, tokenizer, filename, validation_data):
    validation_data_ = validation_data.copy()
    df = pd.DataFrame({"text":sorted(set(validation_data_["less_toxic"].unique()) |
                                     set(validation_data_["more_toxic"].unique()))})
    
    if filename is None:
        preds = predict_cv_fun(cfg, df, tokenizer, text_col="text")
    else:
        preds = predict_fun(cfg, df, tokenizer, filename, text_col="text")

    if np.ndim(preds) > 1:
        df["preds"] = np.mean(preds, axis=1)
    else:
        df["preds"] = preds.reshape(-1)

    validation_data_ = (pd.merge(
        validation_data_, df, left_on="less_toxic", right_on="text", how="left").
        rename(columns={"preds":"less_toxic_preds"}).
        drop("text", axis=1))
    
    validation_data_ = (pd.merge(
        validation_data_, df, left_on="more_toxic", right_on="text", how="left").
        rename(columns={"preds":"more_toxic_preds"}).
        drop("text", axis=1))
    
    return validation_data_


def get_score(validation_data_hat):
    less_toxic, more_toxic = validation_data_hat["less_toxic_preds"], validation_data_hat["more_toxic_preds"]
    return np.mean(more_toxic > less_toxic)


# Train and predict

def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


def train_for_fold(cfg, train_df, valid_df, tokenizer, filename, text_col):

    lightning_datamodule = CompetDataModule(
        cfg=cfg, 
        tokenizer=tokenizer,
        train_df=train_df, 
        valid_df=valid_df, 
        text_col=text_col
        )
    
    lightning_model = CompetModel(cfg=cfg)
    lightning_model.dataset_size = len(train_df)

    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=cfg.exper_model,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    if cfg.early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", 
            min_delta=0.0, 
            patience=8, 
            mode='min', 
        )
        callbacks += [early_stopping]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        deterministic=False,
        accelerator="gpu",
        devices="auto",
        precision=16,
    )

    trainer.fit(lightning_model, datamodule=lightning_datamodule, ckpt_path=cfg.resume_from_checkpoint)
    torch.cuda.empty_cache()


def get_filenames(dirctory):
    listdir = os.listdir(dirctory)
    out_lst = [os.path.splitext(d)[0] for d in listdir]
    return out_lst


def train_cv_fun(cfg, df, tokenizer, text_col=None, validation_data=None, get_oof=True):
    
    oof_df = pd.DataFrame(np.zeros((len(df), len(cfg.target_cols))), columns=cfg.target_cols)

    for i_fold in range(cfg.n_fold):

        if i_fold in cfg.train_fold_num:
            
            filename = f"{cfg.target_data_name}-seed{cfg.seed_num}-fold{i_fold}"
            filelist = get_filenames(cfg.exper_model)

            val_mask = (df["fold"] == i_fold).astype(bool)
            train_df = df[~val_mask].reset_index(drop=True)
            valid_df = df[val_mask].reset_index(drop=True)

            if not filename in filelist:
                print(f"### Start Training Fold={i_fold} ###")
                
                train_for_fold(
                    cfg=cfg, 
                    train_df=train_df, 
                    valid_df=valid_df, 
                    tokenizer=tokenizer, 
                    filename=filename, 
                    text_col=text_col
                    )

            print("### go to validation data score step ###")
            
            if validation_data is not None:
                validation_data_hat = validation_data_hat_fun(cfg, tokenizer, filename, validation_data)
                val_score = get_score(validation_data_hat)
                log = f"{cfg.target_data_name}-seed{cfg.seed_num}-fold{i_fold}: validation data score={val_score:.4f}"
                print(log)
                
            print("### go to validation prediction step ###")
            
            if get_oof:
                preds = predict_fun(
                    cfg=cfg,
                    df=valid_df, 
                    tokenizer=tokenizer, 
                    filename=filename, 
                    text_col=text_col)
                
                oof_df.loc[val_mask] = preds
                return oof_df


def predict_fun(cfg, df, tokenizer, filename, text_col):
    test_dataset = CompetTestDataset(
        cfg=cfg, tokenizer=tokenizer, df=df, text_col=text_col)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=False
        ) 
    
    lightning_model = CompetModel(cfg=cfg).to(cfg.device).eval()
    checkpoint_path = os.path.join(cfg.exper_model, filename + ".ckpt")
    lightning_model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)

    num_targets = len(cfg.target_cols)
    preds = np.zeros((len(df), num_targets))
    fill_start_idx = 0

    for inputs in tqdm(test_dataloader,total=len(test_dataloader)):
    
        for ks, vs in inputs.items():
            inputs[ks] = vs.to(cfg.device)

        with torch.no_grad():
            pred = lightning_model(inputs)
            pred = pred.cpu().numpy()
        
        fill_end_idx = pred.shape[0] + fill_start_idx
        preds[fill_start_idx:fill_end_idx] = pred
        fill_start_idx = fill_end_idx
        
    
    del test_dataset, test_dataloader, lightning_model
    gc.collect()

    return preds


def predict_cv_fun(cfg, df, tokenizer, text_col):
    num_targets = len(cfg.target_cols)
    preds = []
    
    for i_fold in range(cfg.n_fold):
        if i_fold in cfg.train_fold_num:
            filename =f"{cfg.target_data_name}-seed{cfg.seed_num}-fold{i_fold}"
            preds_fold = predict_fun(cfg, df, tokenizer, filename, text_col)
            preds.append(preds_fold)
    
    preds = np.mean(preds, axis=0)  ### fold mean
    return preds


# Create data

def read_csv(filepath, **kwargs):
    if os.path.isdir(filepath):
        filename = filepath.split("/")[-1]
        filepath = os.path.join(filepath, filename)
        
    try:
        csv_data = pd.read_csv(filepath,  **kwargs)
    except:
        csv_data = pd.read_csv(filepath + ".zip",  **kwargs)

    return csv_data


def text_cleaning(text):
    '''
    tips from public codes in Jigsaw competitions
    '''
    template = re.compile(r'https?://\S+|www\.\S+') #website links
    text = template.sub(r'', text)
    
    soup = BeautifulSoup(text, 'lxml') #HTML tags
    only_text = soup.get_text()
    text = only_text
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(' +', ' ', text)
    ipPattern = re.compile('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}') # IP
    text = ipPattern.sub(r'', text)
    bikkuri = re.compile('!')
    text = bikkuri.sub(r' ', text)
    text = text.replace('\n','')
    text = text.replace("\'","")
    text = text.replace("|","")
    text = text.replace("=","")
    text = text.replace("F**K", "FUCK")
    text = text.replace("F__K", "FUCK")
    text = text.replace("f**k", "fuck")
    text = text.replace("f__k", "fuck")
    text = text.replace("f*ck", "fuck")    
    text = text.replace("S$X", "SEX")
    text = text.replace("s$x", "sex")
    text = text.replace(" u ", " you ")
    text = text.replace(" u ", " you ")
    text = text.replace(" U ", " you ")
    text = text.replace(" U ", " you ")
    text = text.replace("YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUUUUUUUUUU", "YOU")
    text = text.strip()
    return text


def text_normalization(s:pd.Series):
    x = s.apply(text_cleaning)
    return x


def jigsaw_01_dataset_proc(cfg):
    '''
    Jigsaw Toxic Comment Classification Challenge:
    text_col --- "comment_text2
    target_cols --- ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    '''
    jigsaw1_train = read_csv(os.path.join(cfg.input_jigsaw_01 , "train.csv"))
    jigsaw1_test = read_csv(os.path.join(cfg.input_jigsaw_01 , "test.csv"))
    jigsaw1_test_label = read_csv(os.path.join(cfg.input_jigsaw_01 , "test_labels.csv"))
    scoring_mask = jigsaw1_test_label["toxic"] != -1
    jigsaw1_test = pd.merge(jigsaw1_test[scoring_mask], jigsaw1_test_label[scoring_mask], on="id", how="left")
    jigsaw1_train = pd.concat([jigsaw1_train, jigsaw1_test], axis=0).reset_index(drop=True)

    return jigsaw1_train


def jigsaw_02_dataset_proc(cfg, cat_threshold=0.5):
    '''
    Jigsaw Unintended Bias in Toxicity Classification:
    text_col --- "comment_text"
    target_cols --- ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    '''
    jigsaw2_data = read_csv(os.path.join(cfg.input_jigsaw_02 , "all_data.csv"), usecols=["id", "comment_text"])
    jigsaw2_labels = read_csv(os.path.join(cfg.input_jigsaw_02 , "toxicity_individual_annotations.csv"))
    jigsaw2_agg_labels = jigsaw2_labels.groupby(["id"]).agg("mean")

    if cat_threshold is not None:
        jigsaw2_agg_labels = pd.DataFrame(
            np.where(jigsaw2_agg_labels >= cat_threshold, 1, 0), 
            index=jigsaw2_agg_labels.index,
            columns=jigsaw2_agg_labels.columns)
    
    jigsaw2_train = pd.merge(jigsaw2_data, jigsaw2_agg_labels, on="id", how="left")
    jigsaw2_train = jigsaw2_train.dropna(axis=0).reset_index(drop=True)
    jigsaw2_train = (jigsaw2_train.
                        rename(columns={"identity_attack":"identity_hate"}).
                        drop(["sexual_explicit", "worker"], axis=1))
    
    return jigsaw2_train

def ruddit_dataset_proc(cfg):
    '''
    Ruddit dataset:
    text_col --- "comment_text"
    target_cols --- "offensiveness_score"
    '''
    ruddit_df = read_csv(os.path.join(cfg.input_ruddit, "Dataset", "ruddit_with_text.csv"))
    ruddit_df = ruddit_df[~ruddit_df["txt"].isin(["[deleted]", "[removed]"])].reset_index(drop=True)
    ruddit_df["comment_text"] = text_normalization(ruddit_df["txt"])
    return ruddit_df.drop("txt", axis=1)


def fold_idx_proc(cfg, df):
    df["fold"] = -1
    y = df[cfg.target_cols].sum(axis=1)
    cv_strategy = KFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed_num)
    for i_fold, (tr_idx, va_idx) in enumerate(cv_strategy.split(X=df, y=y)):
        df.loc[va_idx, "fold"] = i_fold
    
    return df


def custom_jigsaw_dataset_proc(cfg, train_data, validation_data):
    '''
    undersampling from public codes in Jigsaw competitions
    target_cols : ["toxic_score"]
    weighted sum of targets:["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    '''
    train_data["toxic_score"] = train_data[cfg.target_cols].sum(axis=1)
    
    toxic_mask = (train_data["toxic_score"] > 0).astype(bool) # undersampling here
    min_len = np.sum(toxic_mask)

    sampled_data = train_data[train_data["toxic_score"] == 0].sample(n=min_len, random_state=cfg.seed_num)
    train_data = pd.concat([train_data[toxic_mask], sampled_data]).reset_index(drop=True).drop("toxic_score", axis=1)

    val_comment_unq = np.unique(validation_data['less_toxic'].tolist() + validation_data['more_toxic'].tolist())
    duplicate_idx = np.isin(train_data['comment_text'], val_comment_unq)
    train_data = train_data.iloc[~duplicate_idx].reset_index(drop=True)

    return train_data


# MAIN PART

print("### Load Data ###")

tokenizer = tokenizer_fun(Conf_obj)

comments_to_score = read_csv(os.path.join(Conf_obj.input_jigsaw_03, "comments_to_score.csv"))
comments_to_score["text"] = text_normalization(comments_to_score["text"])
sample_submission = read_csv(os.path.join(Conf_obj.input_jigsaw_03, "sample_submission.csv"))



if not Conf_obj.inference_launch_only:

    validation_data = read_csv(os.path.join(Conf_obj.input_jigsaw_03, "validation_data.csv"))

    train_data = jigsaw_01_dataset_proc(cfg=Conf_obj)
    train_data = custom_jigsaw_dataset_proc(cfg=Conf_obj, train_data=train_data, validation_data=validation_data)
    train_data = fold_idx_proc(cfg=Conf_obj, df=train_data)

    train_data["comment_text"] = text_normalization(train_data["comment_text"])
    validation_data["less_toxic"] = text_normalization(validation_data["less_toxic"])
    validation_data["more_toxic"] = text_normalization(validation_data["more_toxic"])


if not Conf_obj.inference_launch_only:

    print("### Training ###")
    
    train_cv_fun(
        cfg=Conf_obj, 
        df=train_data, 
        tokenizer=tokenizer, 
        text_col="comment_text", 
        validation_data=validation_data, 
        get_oof=False
    )


if not Conf_obj.inference_launch_only:
    
    print("### Validation ###")
    
    validation_data_hat = validation_data_hat_fun(
        cfg=Conf_obj, 
        tokenizer=tokenizer, 
        filename=None, 
        validation_data=validation_data
    )
    
    filepath = os.path.join(Conf_obj.exper_pred, "validation_data.csv")
    validation_data.to_csv(filepath, index=False)
    score = get_score(validation_data_hat)
    print(f"validation score = {score:.4f}")


print("### Inference ###")

preds = predict_cv_fun(
    cfg=Conf_obj, 
    df=comments_to_score, 
    tokenizer=tokenizer, 
    text_col="text")

if np.ndim(preds) > 1:
    sub_preds = np.mean(preds, axis=1)
else:
    sub_preds = preds



sample_submission["score"] = sub_preds
filename = "submission.csv"
sample_submission.to_csv(os.path.join(Conf_obj.submission, filename), index=False)

