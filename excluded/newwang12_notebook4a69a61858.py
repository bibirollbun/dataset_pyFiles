"""
Offline inference script for the Kaggle "MAP: Charting Student Math Misunderstandings" competition.

This script reuses the same preprocessing and model architecture that were used during
training in order to ensure that the embeddings computed for the test data are
compatible with those learned for the training data.  It is intended for use
in an offline Kaggle notebook environment where internet access is disabled.

The script performs the following steps:

1. Build the SentenceTransformer backbone from a local HuggingFace checkpoint.  The
   architecture mirrors the training configuration (pooling, dropout, dense
   projection, layer norm) so that the saved weights can be loaded.
2. Load the finetuned weights from ``best_model.bin``.
3. Construct the label bank from the training CSV.  This ensures that all
   categories and misconceptions seen during training are available during
   inference.  The label bank is embedded using the same ``max_seq_length``
   setting as during training (72 tokens).
4. Preprocess each student explanation in the test set using the same
   ``make_input`` function as used for training.  This function lowercases,
   strips LaTeX, and prefixes the explanation with ``"A: "`` by default.
5. Encode the preprocessed test explanations with the model using
   ``model.max_seq_length = 72`` and cosine-normalised embeddings.
6. Compute cosine similarity between each test embedding and every label
   embedding, select the topâ€‘3 labels, and write a submission CSV in the
   required format.

Adjust the ``_*`` constants at the top of the file to point to the correct
locations of your local model folder, checkpoint, training CSV and test CSV on
Kaggle.  See the ``if __name__ == "__main__"`` block for an example.
"""

import os
import json
import gc
from typing import List, Tuple
import math
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
BACKBONE_FOLDER = "mapdataset"
BEST_MODEL_PATH = "MODELMAP/best_model2.bin"
TRAIN_CSV = "mapdataset/train.csv"
TEST_CSV = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
CE_MODEL = "ceversion3"
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BACKBONE_FOLDER = os.getenv("BACKBONE_FOLDER", "/kaggle/input/mapdataset")
BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH", "/kaggle/input/beversion33/best_model3.bin")
TRAIN_CSV       = os.getenv("TRAIN_CSV", "/kaggle/input/mapdataset/train.csv")
TEST_CSV        = os.getenv("TEST_CSV", "/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
CE_MODEL_DIR    = os.getenv("CE_MODEL", "/kaggle/input/ceversion3")
K_BE = 5                            # top-K à¸—à¸µà¹ˆ BE à¸ªà¹ˆà¸‡à¹€à¸‚à¹‰à¸² CE
# à¸Šà¸·à¹ˆà¸­à¹„à¸Ÿà¸¥à¹Œ submission à¸—à¸µà¹ˆà¸ˆà¸°à¹€à¸‹à¸Ÿà¸­à¸­à¸�à¸¡à¸²
OUTPUT_CSV: str = os.environ.get("OUTPUT_CSV", "submission.csv")
INFER_MODE = "CE"
# ---------- à¸„à¹ˆà¸²à¸•à¹ˆà¸²à¸‡ à¹† à¸”à¹‰à¸²à¸™à¸¥à¹ˆà¸²à¸‡à¸„à¸‡à¹€à¸”à¸´à¸¡ ----------
MAX_LEN_MODEL: int = 72        # à¸•à¹‰à¸­à¸‡à¸•à¸£à¸‡à¸�à¸±à¸šà¸•à¸­à¸™à¹€à¸—à¸£à¸™
CE_MAX_LEN: int = 128    # à¸„à¸§à¸²à¸¡à¸¢à¸²à¸§à¹‚à¸—à¹€à¸„à¸™à¸‚à¸­à¸‡ CE à¸•à¸­à¸™à¹€à¸—à¸£à¸™
MAX_LEN_INPUT: int = 192       # pre-truncate à¸�à¹ˆà¸­à¸™à¹€à¸‚à¹‰à¸² tokenizer
ENSURE = False
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------------------
# Preprocessing utilities (mirroring training code)
# -----------------------------------------------------------------------------

import re

def clean_text(
    txt: str,
    lower: bool = False,
    strip_punct: bool = False,
    remove_mathjax: bool = True,
    replace_num: bool = False,
) -> str:
    """Basic cleaning used during training.

    Removes LaTeX math, optionally converts to lowercase, removes punctuation,
    and collapses whitespace.

    Args:
        txt: Input string.
        lower: Whether to convert the text to lowercase.
        strip_punct: Whether to remove punctuation characters.
        remove_mathjax: Whether to remove inline LaTeX expressions denoted by
            dollar signs.
        replace_num: Whether to replace numbers with a ``<NUM>`` token.

    Returns:
        The cleaned string.
    """
    if lower:
        txt = txt.lower()
    if remove_mathjax:
        txt = re.sub(r"\$[^$]*\$", " ", txt)
    if replace_num:
        txt = re.sub(r"\d+(\.\d+)?", "<NUM>", txt)
    if strip_punct:
        txt = re.sub(r"[^\w\s<>]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def make_input(
    row: pd.Series,
    use_q: bool = False,
    use_choice: bool = False,
    join_token: str = "\n",
    prefix_style: str = "label",
    lower_case: bool = False,
    max_len: int = MAX_LEN_INPUT,
    **clean_kw,
) -> str:
    """Compose the input string for the model.

    This mirrors the function used during training.  It concatenates the
    question, multiple choice answer and student explanation with optional
    prefixes.  Only the student explanation is used by default.

    Args:
        row: A pandas Series containing at least the key ``StudentExplanation``.
        use_q: Whether to include the question text.
        use_choice: Whether to include the multiple choice answer.
        join_token: Separator used between segments.
        prefix_style: If ``"label"`` then prefixes segments with ``"Q: "``,
            ``"Choice: "`` and ``"A: "`` as appropriate.  Otherwise uses raw
            text.
        lower_case: Whether to lowercase the text.
        max_len: Maximum number of words considered before truncation.
        **clean_kw: Additional keyword arguments passed to ``clean_text``.

    Returns:
        A preprocessed string ready for tokenisation.
    """
    parts: List[str] = []
    if use_q and "QuestionText" in row:
        q = clean_text(row["QuestionText"], lower_case, **clean_kw)
        parts.append(f"Q: {q}" if prefix_style == "label" else q)
    if use_choice and "MC_Answer" in row:
        c = clean_text(row["MC_Answer"], lower_case, **clean_kw)
        parts.append(f"Choice: {c}" if prefix_style == "label" else c)
    # Always include student explanation
    a = clean_text(row["StudentExplanation"], lower_case, **clean_kw)
    parts.append("query: " + a)
    text = join_token.join(parts)
    # Preâ€‘truncate by word count to reduce overhead; final truncation is done by
    # the tokenizer based on ``MAX_LEN_MODEL``
    words = text.split()
    if len(words) > max_len * 1.5:
        text = " ".join(words[: int(max_len * 1.5)])
    return text


# -----------------------------------------------------------------------------
# Backbone construction (mirroring training code)
# -----------------------------------------------------------------------------

class EmbLayerNorm(torch.nn.Module):
    """Wrap a LayerNorm to operate on the sentence embedding in-place."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, features: dict, **kwargs) -> dict:
        emb = features["sentence_embedding"]  # (B, dim)
        features["sentence_embedding"] = self.norm(emb)
        return features

from sentence_transformers import SentenceTransformer, models


def build_backbone() -> SentenceTransformer:
    # à¹‚à¸«à¸¥à¸”à¹‚à¸¡à¹€à¸”à¸¥à¸�à¸²à¸™à¹�à¸¥à¸° tokenizer à¸ˆà¸²à¸�à¹‚à¸Ÿà¸¥à¹€à¸”à¸­à¸£à¹Œ BACKBONE_FOLDER à¹�à¸šà¸š local_files_only
    transformer = models.Transformer(
        BACKBONE_FOLDER,
        model_args={'local_files_only': True},
        tokenizer_args={'local_files_only': True},
    )
    # à¹€à¸¥à¸·à¸­à¸� pooling à¹�à¸šà¸š mean (à¹„à¸¡à¹ˆà¹ƒà¸Šà¹‰ CLS à¸«à¸£à¸·à¸­ max)
    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=False,
        pooling_mode_mean_tokens=True,
        pooling_mode_max_tokens=False,
    )
    # dropout à¸ªà¸³à¸«à¸£à¸±à¸š sentence embedding
    class EmbDropout(torch.nn.Module):
        def __init__(self, p: float) -> None:
            super().__init__()
            self.p = p
        def forward(self, features: dict, **kwargs) -> dict:
            emb = features["sentence_embedding"]
            features["sentence_embedding"] = torch.nn.functional.dropout(emb, self.p, self.training)
            return features
    dropout = EmbDropout(0.1)

    # dense projection à¹€à¸�à¸·à¹ˆà¸­à¸¥à¸”à¸¡à¸´à¸•à¸´à¸¥à¸‡à¹€à¸«à¸¥à¸·à¸­ 256
    proj_dim = 256
    dense = models.Dense(
        in_features=transformer.get_word_embedding_dimension(),
        out_features=proj_dim,
        activation_function=torch.nn.Tanh(),
    )

    # LayerNorm à¸«à¸¥à¸±à¸‡ dense
    norm = EmbLayerNorm(proj_dim)

    # à¸£à¸§à¸¡à¹‚à¸¡à¸”à¸¹à¸¥à¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”à¹�à¸¥à¹‰à¸§à¸ªà¸£à¹‰à¸²à¸‡ SentenceTransformer
    modules = [transformer, pooling, dropout, dense, norm]
    return SentenceTransformer(modules=modules, device=DEVICE)
# -----------------------------------------------------------------------------
# Label bank construction
# -----------------------------------------------------------------------------

def load_mis_lookup(
    train_csv: str,
    labels: List[str],
    model: SentenceTransformer,
    per_label: int = 50,
    include_question: bool = False,   # à¸–à¹‰à¸²à¸ˆà¸°à¸�à¹ˆà¸§à¸‡à¹‚à¸ˆà¸—à¸¢à¹Œ à¹ƒà¸«à¹‰ True à¹„à¸”à¹‰
    cache_path: str | None = "mis_prototypes_fp16.pt",
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """
    à¸ªà¸£à¹‰à¸²à¸‡ 'à¹‚à¸›à¸£à¹‚à¸•à¹„à¸—à¸›à¹Œ' à¸•à¹ˆà¸­à¸„à¸¥à¸²à¸ªà¸ˆà¸²à¸� StudentExplanation à¸ˆà¸£à¸´à¸‡ à¹† à¹�à¸¥à¹‰à¸§à¸—à¸³ centroid
    à¸„à¸·à¸™ (mis_df, mis_vec) à¹‚à¸”à¸¢ mis_vec à¸¡à¸µ 1 à¹�à¸–à¸§à¸•à¹ˆà¸­à¸„à¸¥à¸²à¸ª à¸•à¸²à¸¡à¸¥à¸³à¸”à¸±à¸šà¹ƒà¸™ labels
    """
    import os, gc
    df = pd.read_csv(train_csv).fillna("")
    df["label"] = df["Category"] + ":" + df["Misconception"]

    # mis_df à¸•à¸²à¸¡à¸¥à¸³à¸”à¸±à¸š labels à¸—à¸µà¹ˆà¸ªà¹ˆà¸‡à¸¡à¸² (à¸•à¹‰à¸­à¸‡à¸•à¸£à¸‡à¸�à¸±à¸š label2id à¸—à¸µà¹ˆà¸™à¸´à¸§à¹ƒà¸Šà¹‰à¸­à¸¢à¸¹à¹ˆ)
    mis_df = pd.DataFrame(labels, columns=["label"])
    mis_df[["Category", "Misconception"]] = mis_df["label"].str.split(pat=":", n=1, expand=True)

    # à¸¥à¸­à¸‡à¹‚à¸«à¸¥à¸” cache
    if cache_path and os.path.exists(cache_path):
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            if data["vec"].shape[0] == len(labels):
                print(f"âœ“ loaded prototype cache: {cache_path}")
                return mis_df, data["vec"].to(DEVICE)
            else:
                print("âš ï¸� prototype cache size mismatch â†’ rebuild")
        except Exception as e:
            print(f"âš ï¸� cache load failed ({e}) â†’ rebuild")

    # à¹€à¸•à¸£à¸µà¸¢à¸¡à¸‚à¹‰à¸­à¸„à¸§à¸²à¸¡à¸•à¸±à¸§à¸­à¸¢à¹ˆà¸²à¸‡à¸•à¹ˆà¸­à¸„à¸¥à¸²à¸ª
    texts_per_label: list[list[str]] = []
    for lbl in labels:
        g = df[df["label"] == lbl]
        if len(g) == 0:
            texts_per_label.append([f"passage: {lbl}"])  # fallback à¸–à¹‰à¸²à¹„à¸¡à¹ˆà¸¡à¸µà¸•à¸±à¸§à¸­à¸¢à¹ˆà¸²à¸‡
            continue
        if per_label < len(g):
            g = g.sample(per_label, random_state=42)
        # à¹ƒà¸Šà¹‰ make_input à¹€à¸�à¸·à¹ˆà¸­ clean à¹ƒà¸«à¹‰à¹€à¸«à¸¡à¸·à¸­à¸™à¸�à¸±à¹ˆà¸‡ query à¹�à¸¥à¹‰à¸§à¹�à¸›à¸¥à¸‡à¹€à¸›à¹‡à¸™ passage:
        tmp = []
        for _, r in g.iterrows():
            t = make_input(r, use_q=include_question, use_choice=False)
            if not t.startswith("query: "):
                t = "query: " + t
            tmp.append("passage: " + t[len("query: "):])
        texts_per_label.append(tmp)

    # flatten à¹�à¸¥à¹‰à¸§ encode
    all_texts, bounds = [], [0]
    for bucket in texts_per_label:
        all_texts.extend(bucket)
        bounds.append(bounds[-1] + len(bucket))

    model.max_seq_length = MAX_LEN_MODEL
    vec_chunks = []
    bs = 256
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        for i in range(0, len(all_texts), bs):
            enc = model.encode(
                all_texts[i:i+bs],
                batch_size=min(bs, len(all_texts) - i),
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=DEVICE,
            )
            vec_chunks.append(enc.cpu())
    all_vec = torch.cat(vec_chunks, dim=0) if vec_chunks else torch.empty((0, model.get_sentence_embedding_dimension()))

    # à¸—à¸³ centroid à¸•à¹ˆà¸­à¸„à¸¥à¸²à¸ª + normalize
    protos = []
    for i in range(len(labels)):
        s, e = bounds[i], bounds[i+1]
        if e > s:
            m = all_vec[s:e].mean(dim=0)
            m = torch.nn.functional.normalize(m, dim=0)
        else:
            # à¸�à¸±à¸™à¸�à¸±à¸‡
            m = model.encode([f"passage: {labels[i]}"], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE).cpu()[0]
        protos.append(m)
    mis_vec = torch.stack(protos, dim=0).to(DEVICE)

    if cache_path:
        # à¹€à¸‹à¸Ÿà¹€à¸›à¹‡à¸™ fp16 à¹ƒà¸«à¹‰à¹€à¸¥à¹‡à¸�
        torch.save({"df": mis_df, "vec": mis_vec.half().cpu()}, cache_path)
        print(f"âœ“ saved prototype cache â†’ {cache_path}")
    return mis_df, mis_vec

# -----------------------------------------------------------------------------
# Inference pipeline
# -----------------------------------------------------------------------------

def run_inference(
    model: SentenceTransformer,
    mis_df: pd.DataFrame,
    mis_vec: torch.Tensor,
    test_csv: str,
    output_csv: str = OUTPUT_CSV,
    batch_size: int = 1024,
    sim_step: int = 20000,
) -> pd.DataFrame:
    """Run inference on the test set and save a submission file.

    Args:
        model: Finetuned SentenceTransformer.
        mis_df: DataFrame containing label strings and their split.
        mis_vec: Tensor of label embeddings.
        test_csv: Path to the test CSV with columns ``row_id`` and
            ``StudentExplanation``.
        output_csv: Path to the output submission CSV.
        batch_size: Batch size for encoding test texts.
        sim_step: Number of test embeddings processed per cosine computation
            chunk.  Reduce this value if your environment has limited RAM/VRAM.

    Returns:
        The submission DataFrame.
    """
    model.eval()
    # Load test data
    test = pd.read_csv(test_csv).fillna("")
    # Preprocess all student explanations using make_input
    texts = [make_input(row) for _, row in test.iterrows()]
    # Encode test explanations in batches
    emb_list: List[torch.Tensor] = []
    model.max_seq_length = MAX_LEN_MODEL
    loader = DataLoader(range(len(texts)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for idxs in tqdm(loader, desc="encode_test", leave=False):
            batch_txt = [texts[i] for i in idxs]
            emb = model.encode(
                batch_txt,
                batch_size=len(batch_txt),
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=DEVICE,
            )
            emb_list.append(emb.cpu())
            del emb; gc.collect()
    test_vec = torch.cat(emb_list).to(DEVICE)
    mis_vec = mis_vec.to(device=test_vec.device, dtype=test_vec.dtype)  # â˜… à¹€à¸�à¸´à¹ˆà¸¡
    del emb_list
    gc.collect()
    # Compute cosine similarities in blocks to avoid memory issues
    cat  = mis_df["Category"].values          # â†� à¸¢à¸�à¸¡à¸²à¸‚à¹‰à¸²à¸‡à¸šà¸™ à¹ƒà¸Šà¹‰à¹„à¸”à¹‰à¸—à¸±à¹‰à¸‡ BE & CE
    misc = mis_df["Misconception"].values
    if INFER_MODE == "BE":
        # ---------- cosine + top-k à¹€à¸«à¸¡à¸·à¸­à¸™à¹€à¸”à¸´à¸¡ ----------
        top_idx: List[np.ndarray] = []
        mis_vec_t = mis_vec.T
        with torch.no_grad():
            for i in tqdm(range(0, len(test_vec), sim_step),
                          desc="cosine", leave=False):
                chunk = test_vec[i : i + sim_step]
                cos   = torch.matmul(chunk, mis_vec_t)        # (b, C)
                idx   = torch.topk(cos, k=3, dim=1).indices.cpu().numpy()
                top_idx.append(idx)
                del cos; gc.collect()
        top_idx = np.vstack(top_idx)
        preds = [" ".join([f"{cat[j]}:{misc[j]}" for j in row]) for row in top_idx]
    elif INFER_MODE == "CE":
        # ---------- rerank à¸”à¹‰à¸§à¸¢ Cross-Encoder ----------
        preds = rerank_with_ce(
            be_vecs=test_vec,            # (N, d) à¸ˆà¸²à¸� BE encode à¹�à¸¥à¹‰à¸§
            mis_vec=mis_vec,
            mis_df=mis_df,
            texts=texts,
            ce_tok=ce_tok,
            ce_mdl=ce_mdl,
            top_k_be=K_BE,
            batch_size=64
        )
    else:
        raise ValueError(f"INFER_MODE '{INFER_MODE}' à¹„à¸¡à¹ˆà¸£à¸¹à¹‰à¸ˆà¸±à¸�à¸ˆà¹‰à¸²")
    submission = pd.DataFrame({
        "row_id": test["row_id"],
        "Category:Misconception": preds,
    })
    submission.to_csv(output_csv, index=False)
    print(f"âœ“ submission.csv saved: {submission.shape} â†’ {output_csv}")
    return submission


def eval_on_split(
    model: SentenceTransformer,
    csv_path: str,
    mis_df: pd.DataFrame,        # â˜… à¹€à¸�à¸´à¹ˆà¸¡
    mis_vec: torch.Tensor,
    split_frac: float = 1.0,
    batch_size: int = 128,
) -> float:
    """
    à¸›à¸£à¸°à¹€à¸¡à¸´à¸™ MAP@3 à¸šà¸™à¹„à¸Ÿà¸¥à¹Œ csv (train/dev) à¹‚à¸”à¸¢à¹ƒà¸Šà¹‰
    pipeline inference à¸ˆà¸£à¸´à¸‡ 1-à¸•à¹ˆà¸­-1 à¸�à¸±à¸šà¸•à¸­à¸™ submit

    Args
    ----
    model       : SentenceTransformer à¸—à¸µà¹ˆà¹‚à¸«à¸¥à¸”à¸™à¹‰à¸³à¸«à¸™à¸±à¸� finetune à¹€à¸£à¸µà¸¢à¸šà¸£à¹‰à¸­à¸¢
    csv_path    : path à¸‚à¸­à¸‡à¹„à¸Ÿà¸¥à¹Œ train/dev à¸—à¸µà¹ˆà¸ˆà¸°à¸§à¸±à¸”
    mis_df      : DataFrame à¸—à¸µà¹ˆà¹„à¸”à¹‰à¸ˆà¸²à¸� load_mis_lookup (à¸¥à¸³à¸”à¸±à¸šà¸•à¸£à¸‡ mis_vec)
    mis_vec     : Tensor (C Ã— d) à¸‚à¸­à¸‡ label-embedding
    split_frac  : à¸ªà¸¸à¹ˆà¸¡à¹ƒà¸Šà¹‰à¸šà¸²à¸‡à¸ªà¹ˆà¸§à¸™à¸‚à¸­à¸‡à¹„à¸Ÿà¸¥à¹Œà¹€à¸�à¸·à¹ˆà¸­à¸¥à¸”à¹€à¸§à¸¥à¸² (0-1)
    batch_size  : batch size à¸•à¸­à¸™ encode
    """
    # ---------- à¹€à¸•à¸£à¸µà¸¢à¸¡à¸‚à¹‰à¸­à¸¡à¸¹à¸¥ ----------
    df = pd.read_csv(csv_path, keep_default_na=False)   # â†� à¸­à¸¢à¹ˆà¸²à¸—à¸³ .fillna("")
    # à¹�à¸—à¸™ "" à¸”à¹‰à¸§à¸¢ "NA" à¹ƒà¸«à¹‰ goldâ€‘label à¸•à¸£à¸‡à¸�à¸±à¸™
    df["Misconception"] = df["Misconception"].replace("", "NA")
    if split_frac < 1.0:
        df = df.sample(frac=split_frac, random_state=42).reset_index(drop=True)

    # ---------- gold-id mapping à¸•à¸£à¸‡à¸�à¸±à¸š mis_vec ----------
    label2id = {lbl: i for i, lbl in enumerate(mis_df["label"])}
    gold = (df["Category"] + ":" + df["Misconception"]).map(label2id)

    ok = gold.notna()                         # à¸�à¸£à¸­à¸‡ label à¸—à¸µà¹ˆà¹„à¸¡à¹ˆà¸¡à¸µà¹ƒà¸™ bank (à¸–à¹‰à¸²à¸¡à¸µ)
    df, gold = df[ok], gold[ok].astype(int).to_numpy()

    # ---------- encode ----------
    texts = [make_input(r) for _, r in df.iterrows()]
    model.max_seq_length = MAX_LEN_MODEL
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=DEVICE,
    )
    mis_vec = mis_vec.to(device=vecs.device, dtype=vecs.dtype)
    # ---------- cosine & MAP@3 ----------
    cos  = vecs @ mis_vec.T                          # (N, C)
    topk = cos.topk(3, dim=1).indices.cpu().numpy()  # (N, 3)
    hits = (topk == gold[:, None])

    rr = hits.argmax(1) + 1          # reciprocal-rank (1..k)
    rr[~hits.any(1)] = 100           # no hit sentinel
    map3 = np.where(rr <= 3, 1 / rr, 0).mean()

    return float(map3)
# -----------------------------------------------------------------------------
# Evaluate with CE rerank (hold-out / dev) ------------------------------------
# -----------------------------------------------------------------------------
@torch.no_grad()
def eval_on_split_ce(
    be_model: SentenceTransformer,
    ce_tok,
    ce_mdl,
    csv_path: str,
    mis_df: pd.DataFrame,
    mis_vec: torch.Tensor,
    top_k_be: int = K_BE,
    split_frac: float = 1.0,
    batch_size_be: int = 128,
    batch_size_ce: int = 32,
) -> float:

    # ---------- load & sample ----------
    df = pd.read_csv(csv_path, keep_default_na=False)
    df["Misconception"] = df["Misconception"].replace("", "NA")
    if split_frac < 1.0:
        df = df.sample(frac=split_frac, random_state=42).reset_index(drop=True)

    label2id = {lbl: i for i, lbl in enumerate(mis_df["label"])}
    gold = (df["Category"] + ":" + df["Misconception"]).map(label2id).to_numpy()

    texts = [make_input(r) for _, r in df.iterrows()]

    # ---------- BE encode ----------
    be_model.max_seq_length = MAX_LEN_MODEL
    vecs = be_model.encode(
        texts,
        batch_size=batch_size_be,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=DEVICE,
        show_progress_bar=True,
    )
    mis_vec = mis_vec.to(device=vecs.device, dtype=vecs.dtype)
    idx_topk = (vecs @ mis_vec.T).topk(top_k_be, dim=1).indices  # (N, K)
    cat  = mis_df["Category"].values
    misc = mis_df["Misconception"].values
    
    # ---------- CE rerank ----------
    preds_idx = np.empty((len(df), 3), dtype=np.int32)

    for start in tqdm(
        range(0, len(vecs), batch_size_ce),
        desc="ce_eval", total=math.ceil(len(vecs)/batch_size_ce), leave=False
    ):
        bs = slice(start, start + batch_size_ce)
        base = idx_topk[bs].cpu().numpy()      # (b, K)
        bsz = base.shape[0]

        # à¹€à¸•à¸£à¸µà¸¢à¸¡ candidate à¹�à¸šà¸š list-of-lists (à¸£à¸­à¸‡à¸£à¸±à¸š L_i à¹„à¸¡à¹ˆà¹€à¸—à¹ˆà¸²à¸�à¸±à¸™)
        cand_lists = []
        pair_q, pair_c = [], []
        for i in range(bsz):
            ans = texts[start + i]
            cand = base[i].tolist()
            gi = int(gold[start + i])
            if ENSURE and gi not in cand:                  # ensure-positive à¸•à¹ˆà¸­à¹�à¸–à¸§
                cand.append(gi)
            cand_lists.append(cand)
            q = ans if ans.startswith("query: ") else ("query: " + ans)
            for j in cand:
                lab = f"{mis_df['Category'].iat[j]}:{mis_df['Misconception'].iat[j]}"
                pair_q.append(q)
                pair_c.append("label: " + lab if not lab.startswith("label: ") else lab)

        enc = ce_tok(
            pair_q, pair_c,
            padding=True,
            truncation="only_first",
            max_length=CE_MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        logits = ce_mdl(**enc).logits.squeeze(-1).cpu().numpy()  # (sum_i L_i,)

        # à¹�à¸šà¹ˆà¸‡ logits à¸�à¸¥à¸±à¸šà¸•à¸²à¸¡à¹�à¸•à¹ˆà¸¥à¸°à¹�à¸–à¸§ à¹�à¸¥à¹‰à¸§à¹€à¸¥à¸·à¸­à¸� top-3 à¸ à¸²à¸¢à¹ƒà¸™à¹�à¸–à¸§à¸™à¸±à¹‰à¸™
        p = 0
        for i, cand in enumerate(cand_lists):
            L = len(cand)
            row_logits = logits[p:p+L]; p += L
            best = row_logits.argsort()[::-1][:3]
            preds_idx[start + i] = np.array([cand[b] for b in best], dtype=np.int32)

    # ---------- MAP@3 ----------
    hits = (preds_idx == gold[:, None])
    rr   = hits.argmax(1) + 1
    rr[~hits.any(1)] = 100
    map3 = np.where(rr <= 3, 1/rr, 0).mean()
    return float(map3)



from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_ce_model(model_dir: str = CE_MODEL_DIR):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to(DEVICE).eval()
    return tok, mdl

@torch.no_grad()
def rerank_with_ce(
        be_vecs: torch.Tensor,
        mis_vec: torch.Tensor,
        mis_df: pd.DataFrame,
        texts: List[str],          # â˜… à¸ªà¹ˆà¸‡à¹€à¸‚à¹‰à¸²à¸¡à¸²
        ce_tok, ce_mdl,
        top_k_be: int = K_BE,
        batch_size: int = 32,
) -> List[str]:
    mis_vec = mis_vec.to(device=be_vecs.device, dtype=be_vecs.dtype)  # â˜… à¹€à¸�à¸´à¹ˆà¸¡
    idx_topk = (be_vecs @ mis_vec.T).topk(top_k_be, dim=1).indices
    cat  = mis_df["Category"].values
    misc = mis_df["Misconception"].values

    preds = []
    for start in tqdm(range(0, len(be_vecs), batch_size), desc="ce_rerank"):
        bs = slice(start, start + batch_size)
        these = idx_topk[bs].cpu().numpy()   # (b, K)
        bsz = these.shape[0]

        pair_q, pair_c = [], []
        for i in range(bsz):
            ans = texts[start + i]
            q = ans if ans.startswith("query: ") else ("query: " + ans)
            for j in these[i]:
                lab = f"{mis_df['Category'].iat[j]}:{mis_df['Misconception'].iat[j]}"
                pair_q.append(q)
                pair_c.append("label: " + lab if not lab.startswith("label: ") else lab)

        enc = ce_tok(
            pair_q, pair_c,
            padding=True,
            truncation="only_first",
            max_length=CE_MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)

        flat = ce_mdl(**enc).logits.squeeze(-1).cpu().numpy()    # (b*K,)
        p = 0
        for i in range(bsz):
            L = len(these[i])
            row_logits = flat[p:p+L]; p += L
            top = row_logits.argsort()[::-1][:3]
            labels = [f"{mis_df['Category'].iat[these[i][t]]}:{mis_df['Misconception'].iat[these[i][t]]}"
                      for t in top]
            preds.append(" ".join(labels))

    return preds
# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def build_label_bank(train_csv: str) -> List[str]:
    df = pd.read_csv(train_csv, keep_default_na=False)
    df["Misconception"] = df["Misconception"].replace("", "NA")
    labels = sorted((df["Category"] + ":" + df["Misconception"]).unique())
    return labels
if __name__ == "__main__":
    assert os.path.isfile(BEST_MODEL_PATH)

    # 1) à¹‚à¸«à¸¥à¸” backbone + weight
    model = build_backbone()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
    model = model.to(DEVICE)
    print(f"âœ“ Model loaded from {BEST_MODEL_PATH}")
    if INFER_MODE == "CE":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        ce_tok = AutoTokenizer.from_pretrained(CE_MODEL_DIR, local_files_only=True)
        ce_mdl = AutoModelForSequenceClassification.from_pretrained(
            CE_MODEL_DIR, local_files_only=True
        ).to(DEVICE).eval()
        print(f"âœ“ CE loaded from {CE_MODEL_DIR}")
    else:
        ce_tok = ce_mdl = None
    # 2) à¸ªà¸£à¹‰à¸²à¸‡â€‘encode label bank ("":NA)
    labels = build_label_bank(TRAIN_CSV)
    mis_df, mis_vec = load_mis_lookup(TRAIN_CSV, labels, model,
                                  per_label=50,
                                  include_question=False,
                                  cache_path="mis_prototypes_fp16.pt")
    print(f"âœ“ Built label bank with {len(labels)} labels")

    # 3) à¸£à¸²à¸¢à¸‡à¸²à¸™à¸ªà¸–à¸´à¸•à¸´ 'NA' vs blank
    stats_df = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    na_str_cnt  = (stats_df["Misconception"] == "NA").sum()
    blank_cnt   = ((stats_df["Misconception"] == "") | stats_df["Misconception"].isna()).sum()
    total_rows  = len(stats_df)
    print(f'"NA" string   : {na_str_cnt:,}')
    print(f'blank / NaN   : {blank_cnt:,}')
    print(f'total rows    : {total_rows:,}')
    print(f'NA string %   : {na_str_cnt / total_rows:.2%}')
    print(f'blank/NaN %   : {blank_cnt  / total_rows:.2%}')

    # 4) à¸•à¸£à¸§à¸ˆ delta à¸™à¹‰à¸³à¸«à¸™à¸±à¸� (debug)
    mean0 = next(build_backbone().parameters()).abs().mean().item()  # backbone fresh
    mean1 = next(model.parameters()).abs().mean().item()
    print(f"Î”weight (absâ€‘mean diff) = {abs(mean1 - mean0):.6f}")

    # 5) infer & à¸ªà¸£à¹‰à¸²à¸‡ submission
    run_inference(model, mis_df, mis_vec, TEST_CSV, OUTPUT_CSV)

    # 6) à¹�à¸ªà¸”à¸‡à¸«à¸±à¸§à¹„à¸Ÿà¸¥à¹Œ submission & sample
    my_sub = pd.read_csv(OUTPUT_CSV)
    print("\n[My submission.head()]")
    print(my_sub.head())

    sample_sub = pd.read_csv(
        "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
    )
    print("\n[Sample submission.head()]")
    print(sample_sub.head())

    # 7) quick devâ€‘split 30% MAP@3
    if INFER_MODE == "CE":
        map_train = eval_on_split_ce(   # à¹€à¸‚à¸µà¸¢à¸™à¸Ÿà¸±à¸‡à¸�à¹Œà¸Šà¸±à¸™ clone à¸‡à¹ˆà¸²à¸¢ à¹†
            model, ce_tok, ce_mdl, TRAIN_CSV,
            mis_df, mis_vec, split_frac=0.30
        )
    else:
        map_train = eval_on_split(
            model, TRAIN_CSV, mis_df, mis_vec, split_frac=0.30
        )
    print(f"ğŸ§ª  MAP@3 on TRAINâ€‘split â‰ˆ {map_train:.4f}")

