#Inference
https://www.kaggle.com/code/harshvardhanchand/lb-805-cross-encoder-inference



import os, sys, json, time, re, random, subprocess, logging, glob
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder, InputExample
os.environ["WANDB_DISABLED"] = "true"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("ce-train")


@dataclass
class TrainConfig:
    model_name: str = "microsoft/deberta-v3-small" # use large
    n_splits: int = 2 # make it 5
    epochs: int = 1 # make it 2
    max_length: int = 512
    batch_size_train: int = 8 
    pred_batch_size_val: int = 64
    seed: int = 42
    use_amp: bool = True
    top_k: int = 3

    enable_cleaning: bool = True
    url_token: str = "<URL>"
    email_token: str = "<EMAIL>"
    phone_token: str = "<PHONE>"

    enable_rule_dropout: bool = True
    rule_dropout_p: float = 0.10

    
    use_fast_tokenizer: bool = False  
    lr: float = 1e-5
    warmup_frac: float = 0.06
    scheduler: str = "WarmupCosine"

CFG = TrainConfig()




WORK_DIR  = "/kaggle/working" 
os.makedirs(WORK_DIR, exist_ok=True)

RUN_DIR = os.path.join(WORK_DIR, time.strftime("ce_deberta_v3_%Y%m%d_%H%M%S"))
os.makedirs(RUN_DIR, exist_ok=True)
logger.info(f"Outputs -> {RUN_DIR}")


DATASET_PATH = "/kaggle/input/jigsaw-agile-community-rules"
TRAIN_PATH = f"{DATASET_PATH}/train.csv" 
logger.info(f"TRAIN_PATH: {TRAIN_PATH}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)


_URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[ -]?)?(?:\(?\d{2,4}\)?[ -]?)?\d{3,4}[ -]?\d{4})")
_WORD_RE  = re.compile(r"\w+")

PROTECT = {"no","not","never","prohibited","forbidden","allowed","disallowed","ban","banned","must","mustn't","mustn’t"}

def _collapse_ws(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def clean_post(text: str) -> str:
    s = text or ""
    if CFG.enable_cleaning:
        s = _URL_RE.sub(CFG.url_token, s)
        s = _EMAIL_RE.sub(CFG.email_token, s)
        s = _PHONE_RE.sub(CFG.phone_token, s)
    return _collapse_ws(s)

def clean_rule(text: str) -> str:
    return _collapse_ws(text or "")

def rule_word_dropout(text: str, p: float) -> str:
    if not CFG.enable_rule_dropout or p <= 0: 
        return text
    toks = re.findall(r"\w+|\W+", text)
    out = []
    for tok in toks:
        if _WORD_RE.fullmatch(tok):
            if tok.lower() in PROTECT:
                out.append(tok); continue
            if random.random() < p:
                continue
        out.append(tok)
    return _collapse_ws("".join(out))

def bullets(title: str, items: List[str]) -> str:
    items = [i for i in items if isinstance(i, str) and i.strip()]
    return (title + "\n" + "\n".join(f"- {i}" for i in items) + "\n") if items else ""

def build_prompt(row: Dict[str, Any], for_train: bool = False) -> str:
    rule_text = clean_rule(row.get("rule", ""))
    if for_train and CFG.enable_rule_dropout:
        rule_text = rule_word_dropout(rule_text, CFG.rule_dropout_p)

    def cpost(x): return clean_post(x) if CFG.enable_cleaning else (x or "")

    pos1 = cpost(row.get("positive_example_1",""))
    pos2 = cpost(row.get("positive_example_2",""))
    neg1 = cpost(row.get("negative_example_1",""))
    neg2 = cpost(row.get("negative_example_2",""))
    body = cpost(row.get("body",""))
    subreddit = (row.get("subreddit","") or "").strip()

    ctx = (
        f"Rule:\n{rule_text}\n\n"
        f"{bullets('Positive examples:', [pos1, pos2])}"
        f"{bullets('Negative examples:', [neg1, neg2])}"
    )
    sub = f"[subreddit: {subreddit}]\n" if subreddit else ""
    post = f"[QUERY POST]\n{sub}{body}"
    return ctx + "\n" + post + "\n\nTask: Does the post violate the rule? Only respond Yes/No"

def df_to_pairs(df: pd.DataFrame, for_train: bool, add_aug: bool) -> List[InputExample]:
    pairs = []
    for _, r in df.iterrows():
        row = {
            "rule": r["rule"],
            "positive_example_1": r.get("positive_example_1",""),
            "positive_example_2": r.get("positive_example_2",""),
            "negative_example_1": r.get("negative_example_1",""),
            "negative_example_2": r.get("negative_example_2",""),
            "subreddit": r.get("subreddit",""),
            "body": r["body"],
        }
        t = build_prompt(row, for_train=for_train)
        y = float(r["rule_violation"]) if "rule_violation" in r else 0.0
        pairs.append(InputExample(texts=[t, ""], label=y))

        if for_train and add_aug:
            if row["positive_example_1"]:
                rr = row.copy(); rr["body"] = row["positive_example_1"]
                pairs.append(InputExample(texts=[build_prompt(rr, True), ""], label=1.0))
            if row["positive_example_2"]:
                rr = row.copy(); rr["body"] = row["positive_example_2"]
                pairs.append(InputExample(texts=[build_prompt(rr, True), ""], label=1.0))
            if row["negative_example_1"]:
                rr = row.copy(); rr["body"] = row["negative_example_1"]
                pairs.append(InputExample(texts=[build_prompt(rr, True), ""], label=0.0))
            if row["negative_example_2"]:
                rr = row.copy(); rr["body"] = row["negative_example_2"]
                pairs.append(InputExample(texts=[build_prompt(rr, True), ""], label=0.0))
    return pairs

def compute_column_avg_auc(df_eval: pd.DataFrame) -> Dict[str, Any]:
    per_rule = {}
    for rule, g in df_eval.groupby("rule"):
        y = g["label"].values
        p = g["pred"].values
        if len(np.unique(y)) < 2:
            continue
        per_rule[rule] = roc_auc_score(y, p)
    macro = float(np.mean(list(per_rule.values()))) if per_rule else float("nan")
    return {"per_rule": per_rule, "column_avg_auc": macro}


if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Could not find train.csv at: {TRAIN_PATH}")

logger.info(f"Loading train: {TRAIN_PATH}")
df = pd.read_csv(TRAIN_PATH)
needed = {"body","rule","rule_violation"}
if not needed.issubset(df.columns):
    raise ValueError(f"train.csv must contain columns {needed}, found {set(df.columns)}")

df["rule_violation"] = df["rule_violation"].astype(int)
logger.info(f"Rows: {len(df)} | Unique rules: {df['rule'].nunique()} | Pos rate: {df['rule_violation'].mean():.3f}")


tok = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=CFG.use_fast_tokenizer)



with open(os.path.join(RUN_DIR, "run_config.json"), "w") as f:
    json.dump(asdict(CFG), f, indent=2)


skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
y = df["rule_violation"].values

oof_parts = []
fold_paths: List[str] = []
fold_scores: List[List[float]] = []


for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), 1):
    logger.info(f"========== Fold {fold}/{CFG.n_splits} ==========")
    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)

    train_pairs = df_to_pairs(df_tr, for_train=True, add_aug=True)
    val_pairs   = df_to_pairs(df_va, for_train=False, add_aug=False)

    model = CrossEncoder(
        CFG.model_name,
        num_labels=1,
        max_length=CFG.max_length,
        tokenizer_args={"use_fast": CFG.use_fast_tokenizer},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    
    try:
        model.model.gradient_checkpointing_enable()
    except Exception:
        pass

    train_loader = DataLoader(train_pairs, shuffle=True, batch_size=CFG.batch_size_train)
    total_steps  = len(train_loader) * CFG.epochs
    warmup_steps = int(total_steps * CFG.warmup_frac)
    logger.info(f"Train steps: {total_steps} | Warmup: {warmup_steps}")

    model.fit(
        train_dataloader=train_loader,
        epochs=CFG.epochs,
        warmup_steps=warmup_steps,
        output_path=None,
        show_progress_bar=True,
        scheduler=CFG.scheduler,
        optimizer_params={"lr": CFG.lr},
        use_amp=CFG.use_amp,
    )

    model.model.eval()

    @torch.no_grad()
    def batched_predict_probs(m, prompts, bs=CFG.pred_batch_size_val, desc="Scoring"):
        out = []
        for i in tqdm(range(0, len(prompts), bs), desc=desc):
            logits = m.predict(prompts[i:i+bs])  # raw logits
            probs = torch.sigmoid(torch.tensor(logits)).numpy().tolist()
            out.extend(probs)
        return np.array(out, dtype=float)

    val_prompts = [[vp.texts[0], ""] for vp in val_pairs]
    val_probs = batched_predict_probs(model, val_prompts, bs=CFG.pred_batch_size_val, desc=f"Fold {fold} validating")

    df_eval = pd.DataFrame({
        "rule": df_va["rule"].values,
        "label": df_va["rule_violation"].astype(int).values,
        "pred":  val_probs
    })
    metrics = compute_column_avg_auc(df_eval)
    col_auc = metrics["column_avg_auc"]
    logger.info("Per-rule AUCs: " + json.dumps({k: round(v,4) for k,v in metrics["per_rule"].items()}, indent=2))
    logger.info(f"Column-averaged AUC: {col_auc:.5f}")

    
    fold_eval_path = os.path.join(RUN_DIR, f"fold{fold}_eval.csv")
    df_eval.to_csv(fold_eval_path, index=False)
    logger.info(f"Saved fold eval -> {fold_eval_path}")

    
    df_eval["fold"] = fold
    oof_parts.append(df_eval)

    
    fold_dir = os.path.join(RUN_DIR, f"model_fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    model.model.half()
    model.model.save_pretrained(fold_dir, safe_serialization=True)
    tok.save_pretrained(os.path.join(fold_dir, "tokenizer"))
    logger.info(f"Saved fold model -> {fold_dir}")

    fold_paths.append(fold_dir)
    fold_scores.append([fold, float(col_auc)])

    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


oof = pd.concat(oof_parts, axis=0, ignore_index=True)
oof_metrics = compute_column_avg_auc(oof)
logger.info("========== OOF Results ==========")
logger.info("Per-rule AUCs: " + json.dumps({k: round(v,4) for k,v in oof_metrics["per_rule"].items()}, indent=2))
logger.info(f"Column-averaged AUC: {oof_metrics['column_avg_auc']:.5f}")

oof_path = os.path.join(RUN_DIR, "oof_predictions.csv")
oof.to_csv(oof_path, index=False)
with open(os.path.join(RUN_DIR, "oof_metrics.json"), "w") as f:
    json.dump(oof_metrics, f, indent=2)

with open(os.path.join(RUN_DIR, "fold_scores.json"), "w") as f:
    json.dump({"fold_scores": fold_scores}, f, indent=2)


fold_scores_sorted = sorted(fold_scores, key=lambda x: x[1], reverse=True)
topk = fold_scores_sorted[:CFG.top_k]
topk_paths = [os.path.join(RUN_DIR, f"model_fold{fold}") for fold, _ in topk]

with open(os.path.join(RUN_DIR, "top_k.json"), "w") as f:
    json.dump({"top_k": topk}, f, indent=2)
with open(os.path.join(RUN_DIR, "top_k_paths.txt"), "w") as f:
    f.write("\n".join(topk_paths) + "\n")

logger.info(f"Top-{CFG.top_k} folds: {topk}")
logger.info("Training complete.")


