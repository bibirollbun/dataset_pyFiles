# -*- coding: utf-8 -*-

import os
import gc
import re
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig


# ====================================================
# CFG
# ====================================================
class CFG:
    debug = False
    num_workers = 0
    model = "/kaggle/input/deberta-v3/deberta_base_cache"
    batch_size = 8
    max_len = 512
    seed = 42
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]


# ====================================================
# Utils
# ====================================================
def get_logger(filename='inference'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=CFG.seed)


# ====================================================
# Model
# ====================================================
class DebertaForTokenBinary(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
        
    def forward(self, **batch):
        labels = batch.pop("labels", None)
        out = self.backbone(**batch)
        logits = self.classifier(self.dropout(out.last_hidden_state)).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {"loss": loss, "logits": logits}


# ====================================================
# Dataset
# ====================================================
class TestDataset(Dataset):
    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, item):
        inputs = self.tokenizer(
            str(self.feature_texts[item]),
            str(self.pn_historys[item]), 
            add_special_tokens=True,
            max_length=CFG.max_len,
            padding="max_length",
            return_offsets_mapping=False,
            truncation="only_second"
        )
        return {k: torch.tensor(v, dtype=torch.long) for k, v in inputs.items()}


# ====================================================
# Helper functions
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    
    for step, inputs in enumerate(tk0):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            y_preds = outputs["logits"].sigmoid()
        preds.append(y_preds.to('cpu').numpy())
    
    predictions = np.concatenate(preds)
    return predictions

def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(
            text, 
            add_special_tokens=True,
            max_length=CFG.max_len,
            padding="max_length", 
            return_offsets_mapping=True
        )
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            if start < len(results[i]) and end <= len(results[i]):
                results[i][start:end] = pred
    return results

def get_predictions_from_char_probs(char_probs, threshold=0.5):
    predictions = []
    for char_prob in char_probs:
        prediction = []
        inside = False
        start_idx = 0
        
        for i, prob in enumerate(char_prob):
            if prob >= threshold and not inside:
                start_idx = i
                inside = True
            elif prob < threshold and inside:
                prediction.append(f"{start_idx} {i}")
                inside = False
                
        # Handle case where span continues to the end
        if inside:
            prediction.append(f"{start_idx} {len(char_prob)}")
            
        predictions.append(" ".join(prediction))
    return predictions

def find_optimal_threshold(char_probs, ground_truth_locations=None):
    """
    å°‹æ‰¾æœ€ä½³é–¾å€¼ - å¦‚æ�œæ²’æœ‰ground truthï¼Œå°±è¼¸å‡ºä¸�å�Œé–¾å€¼çš„çµ�æ�œä¾›å�ƒè€ƒ
    """
    thresholds = np.arange(0.45, 0.56, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    LOGGER.info("=== Threshold Search ===")
    
    for th in thresholds:
        predictions = get_predictions_from_char_probs(char_probs, threshold=th)
        
        if ground_truth_locations is not None:
            # å¦‚æ�œæœ‰ground truthï¼Œè¨ˆç®—å¯¦éš›F1åˆ†æ•¸
            score = compute_f1_score(ground_truth_locations, predictions)
        else:
            # å¦‚æ�œæ²’æœ‰ground truthï¼Œä½¿ç”¨é �æ¸¬æ•¸é‡�ä½œç‚ºå�ƒè€ƒæŒ‡æ¨™
            total_predictions = sum(1 for pred in predictions if pred.strip())
            avg_prediction_length = np.mean([len(pred) for pred in predictions if pred.strip()])
            score = total_predictions / len(predictions) * 0.5 + min(avg_prediction_length / 20, 0.5)
        
        LOGGER.info(f"th: {th:.2f}  score: {score:.5f}")
        
        if score > best_score:
            best_score = score
            best_threshold = th
    
    LOGGER.info(f"best_th: {best_threshold:.2f}  score: {best_score:.5f}")
    LOGGER.info("=" * 30)
    
    return best_threshold

def compute_f1_score(ground_truth, predictions):
    """
    è¨ˆç®—F1åˆ†æ•¸ (å¦‚æ�œæœ‰ground truthçš„è©±)
    """
    # é€™æ˜¯ä¸€å€‹ç°¡åŒ–çš„F1è¨ˆç®—ï¼Œå¯¦éš›æ¯”è³½ä¸­æœƒæ›´è¤‡é›œ
    tp = fp = fn = 0
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set()
        pred_set = set()
        
        # è§£æ��ground truth spans
        if gt and gt.strip():
            for span in gt.split(';'):
                if span.strip():
                    try:
                        start, end = map(int, span.strip().split())
                        gt_set.update(range(start, end))
                    except:
                        pass
        
        # è§£æ��prediction spans  
        if pred and pred.strip():
            for span in pred.split(';'):
                if span.strip():
                    try:
                        start, end = map(int, span.strip().split())
                        pred_set.update(range(start, end))
                    except:
                        pass
        
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
    
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return f1


# ====================================================
# main
# ====================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # ====================================================
    # Load test data
    # ====================================================
    LOGGER.info("Loading test data...")
    test = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/test.csv')
    features = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/features.csv')
    patient_notes = pd.read_csv('/kaggle/input/nbme-score-clinical-patient-notes/patient_notes.csv')
    
    test = test.merge(features, on=['feature_num', 'case_num'], how='left')
    test = test.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
    LOGGER.info(f"Test data shape: {test.shape}")

    # ====================================================
    # Load tokenizer
    # ====================================================
    LOGGER.info("Loading tokenizer...")
    try:
        if os.path.exists(CFG.model):
            LOGGER.info(f"Loading tokenizer from local path: {CFG.model}")
            tokenizer = AutoTokenizer.from_pretrained(CFG.model)
        else:
            LOGGER.info("Local path not found, using online model...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
            CFG.model = "microsoft/deberta-base"
        LOGGER.info("âœ… Tokenizer loaded successfully")
    except Exception as e:
        LOGGER.error(f"â�Œ Failed to load tokenizer: {e}")
        raise

    # ====================================================
    # Create dataset and dataloader
    # ====================================================
    test_dataset = TestDataset(tokenizer, test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True, 
        drop_last=False
    )

    # ====================================================
    # Model inference
    # ====================================================
    predictions = []
    LOGGER.info("Starting inference...")
    
    for fold in CFG.trn_fold:
        LOGGER.info(f"Processing fold {fold}...")
        
        try:
            model_path = f'/kaggle/input/nbme-train-v3/nbme_ckpt/fold{fold}.pt'
            if not os.path.exists(model_path):
                LOGGER.error(f"â�Œ Model file not found: {model_path}")
                continue
                
            # è¼‰å…¥æ¨¡å�‹
            model = DebertaForTokenBinary(CFG.model)
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            LOGGER.info(f"âœ… Model loaded for fold {fold}")
            
            # æ�¨è«–
            prediction = inference_fn(test_loader, model, device)
            predictions.append(prediction)
            LOGGER.info(f"âœ… Fold {fold} inference completed")
            
            del model, checkpoint, prediction
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            LOGGER.error(f"â�Œ Error in fold {fold}: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            continue

    if len(predictions) == 0:
        raise RuntimeError("â�Œ No models were successfully loaded!")

    # ====================================================
    # Ensemble and post-processing
    # ====================================================
    LOGGER.info("Averaging predictions...")
    predictions = np.mean(predictions, axis=0)

    LOGGER.info("Post processing...")
    char_probs = get_char_probs(test['pn_history'].values, predictions, tokenizer)
    
    # ğŸ”¥ æ–°å¢�ï¼šé–¾å€¼æ�œç´¢
    best_threshold = find_optimal_threshold(char_probs, ground_truth_locations=None)
    
    predictions = get_predictions_from_char_probs(char_probs, threshold=best_threshold)

    # ====================================================
    # Create submission
    # ====================================================
    LOGGER.info("Creating submission...")
    test['location'] = predictions
    submission = test[['id', 'location']].copy()
    submission.to_csv('submission.csv', index=False)
    LOGGER.info("âœ… submission.csv saved")
    LOGGER.info(f"Submission shape: {submission.shape}")
    print(submission.head())

if __name__ == '__main__':
    main()

