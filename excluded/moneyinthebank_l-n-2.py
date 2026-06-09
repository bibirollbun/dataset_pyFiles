import pathlib, os, pandas as pd

# Kaggle tá»± mount dá»¯ liá»‡u cá»§a cuá»™c thi vÃ o thÆ° má»¥c nÃ y
DATA_DIR = pathlib.Path("/kaggle/input/financial-chat-bot")
assert DATA_DIR.exists(), "ChÆ°a tháº¥y thÆ° má»¥c competition â€“ hÃ£y kiá»ƒm tra láº¡i Add-on Data"

# Xem nhanh cÃ¡c file bÃªn trong
list(DATA_DIR.iterdir())



# Cell 2: Locate the test_question file and load it
import pandas as pd

# TÃ¬m tá»‡p cÃ³ tÃªn báº¯t Ä‘áº§u báº±ng "test_question"
test_files = list(DATA_DIR.rglob("test_question.*"))
assert len(test_files) > 0, "KhÃ´ng tÃ¬m tháº¥y file test_question.* â€“ kiá»ƒm tra láº¡i dataset"
test_path = test_files[0]          # náº¿u cÃ³ nhiá»�u, láº¥y file Ä‘áº§u tiÃªn
print("Ä�ang Ä‘á»�c:", test_path)

# Ä�á»�c tÃ¹y theo pháº§n má»Ÿ rá»™ng
if test_path.suffix in {".xlsx", ".xls"}:
    df_test = pd.read_excel(test_path)
else:  # giáº£ sá»­ .csv
    df_test = pd.read_csv(test_path)

print(f"Sá»‘ cÃ¢u há»�i: {len(df_test)}")
display(df_test.head())



# Cell 3: quick sanity-check vÃ  chuáº©n hoÃ¡ tÃªn cá»™t

# 1) ThÃ´ng tin tá»•ng quÃ¡t
df_test.info()

# 2) Kiá»ƒm tra giÃ¡ trá»‹ thiáº¿u
print("\nSá»‘ Ã´ NaN theo cá»™t:")
print(df_test.isna().sum())

# 3) Ä�á»•i tÃªn cá»™t 'inputs' â†’ 'Question' (Ä‘á»ƒ thá»‘ng nháº¥t vá»� sau)
if 'inputs' in df_test.columns:
    df_test = df_test.rename(columns={'inputs': 'Question'})

# 4) Kiá»ƒm tra trÃ¹ng ID (khÃ´ng nÃªn cÃ³)
duplicates = df_test.duplicated(subset=['ID']).sum()
print(f"\nSá»‘ ID trÃ¹ng láº·p: {duplicates}")
assert duplicates == 0, "CÃ³ ID bá»‹ trÃ¹ng â€“ cáº§n xá»­ lÃ½!"

# Hiá»ƒn thá»‹ 5 dÃ²ng Ä‘áº§u sau khi Ä‘á»•i tÃªn
display(df_test.head())



# Cell 4: táº¡o cá»™t `output` táº¡m thá»�i vÃ  xem trÆ°á»›c

def dummy_answer(question: str) -> str:
    """
    Placeholder tráº£ lá»�i máº·c Ä‘á»‹nh.
    Sau nÃ y báº¡n sáº½ thay báº±ng hÃ m RAG/LLM tháº­t.
    """
    return "Xin chÃ o! MÃ¬nh sáº½ cáº­p nháº­t cÃ¢u tráº£ lá»�i chi tiáº¿t sá»›m."

# GÃ¡n cÃ¢u tráº£ lá»�i cho má»�i hÃ ng
df_test["output"] = df_test["Question"].apply(dummy_answer)

# Kiá»ƒm tra 5 dÃ²ng Ä‘áº§u
display(df_test[["ID", "Question", "output"]].head())



# Cell 5: lÆ°u submission.csv vÃ  xem trÆ°á»›c

import pathlib

WORK_DIR = pathlib.Path("/kaggle/working")
WORK_DIR.mkdir(exist_ok=True)

sub_path = WORK_DIR / "submission.csv"
df_test[["ID", "output"]].to_csv(sub_path, index=False)

print(f"âœ… Ä�Ã£ ghi submission tá»›i: {sub_path}")

# Xem 5 dÃ²ng Ä‘áº§u Ä‘á»ƒ cháº¯c cháº¯n
import pandas as pd
display(pd.read_csv(sub_path).head())



# Cell 6 â€“ liá»‡t kÃª cáº¥u trÃºc thÆ° má»¥c news
from pathlib import Path
from collections import Counter

NEWS_DIR = DATA_DIR / "news"
assert NEWS_DIR.exists(), "ThÆ° má»¥c news khÃ´ng tá»“n táº¡i â€“ kiá»ƒm tra láº¡i!"

# Ä�áº¿m sá»‘ file theo pháº§n má»Ÿ rá»™ng
ext_counter = Counter(p.suffix.lower() for p in NEWS_DIR.rglob("*") if p.is_file())
print("ğŸ“„ Sá»‘ file theo loáº¡i:", ext_counter)

# Hiá»ƒn thá»‹ 10 file Ä‘áº§u tiÃªn Ä‘á»ƒ hÃ¬nh dung
sample_files = [str(p) for p in NEWS_DIR.rglob("*") if p.is_file()][:10]
for f in sample_files:
    print("â€¢", f)



# Cell 7 â€“ Ä‘á»�c news.csv vÃ  xem cáº¥u trÃºc
import pandas as pd

news_csv_path = NEWS_DIR / "news.csv"
assert news_csv_path.exists(), f"KhÃ´ng tÃ¬m tháº¥y {news_csv_path}"

df_news_raw = pd.read_csv(news_csv_path, on_bad_lines="skip")
print(f"ğŸ”¹ Sá»‘ dÃ²ng Ä‘á»�c Ä‘Æ°á»£c: {len(df_news_raw)}")
display(df_news_raw.head())

# Kiá»ƒm tra danh sÃ¡ch cá»™t Ä‘á»ƒ ta chuáº©n hoÃ¡ á»Ÿ bÆ°á»›c káº¿ tiáº¿p
print("\nCÃ¡c cá»™t hiá»‡n cÃ³:", list(df_news_raw.columns))



# Cell 8 â€“ chá»�n cá»™t cáº§n thiáº¿t, lÃ m sáº¡ch ná»™i dung, xoÃ¡ báº£n ghi quÃ¡ ngáº¯n / trÃ¹ng
import re, unicodedata
import pandas as pd

# 1) Giá»¯ cÃ¡c cá»™t quan trá»�ng & Ä‘á»•i tÃªn cho ngáº¯n gá»�n
df_news = df_news_raw.rename(columns={
    "content": "content",
    "source":  "url",
    "news_date": "date"
})[["content", "url", "date"]].copy()

# 2) HÃ m lÃ m sáº¡ch ná»™i dung
def clean_text(txt: str) -> str:
    if pd.isna(txt): return ""
    txt = str(txt)
    txt = re.sub(r"<[^>]+>", " ", txt)               # bá»� HTML tags
    txt = unicodedata.normalize("NFKC", txt)         # chuáº©n Unicode
    txt = re.sub(r"\s+", " ", txt).strip()           # gá»™p khoáº£ng tráº¯ng
    return txt

df_news["content"] = df_news["content"].apply(clean_text)

# 3) Loáº¡i bá»� báº£n ghi content quÃ¡ ngáº¯n (< 30 kÃ½ tá»±) vÃ  trÃ¹ng láº·p
df_news = df_news[df_news["content"].str.len() > 30]
df_news = df_news.drop_duplicates(subset=["content"])

print("âœ… Sau lÃ m sáº¡ch: ", len(df_news), "bÃ i viáº¿t")
display(df_news.head(3))



# Cell 9 â€“ chia má»—i bÃ i viáº¿t thÃ nh cÃ¡c chunk ~300 tá»«
from nltk.tokenize import word_tokenize
import nltk, math
from tqdm.auto import tqdm
import pandas as pd

# Ä‘áº£m báº£o tokenizer Ä‘Ã£ sáºµn sÃ ng
nltk.download('punkt', quiet=True)

CHUNK_SIZE = 300       # sá»‘ tá»« trong 1 chunk
CHUNK_OVERLAP = 50     # chá»“ng láº¥n 50 tá»« Ä‘á»ƒ khÃ´ng máº¥t ngá»¯ cáº£nh
MIN_WORDS   = 50       # bá»� chunk quÃ¡ ngáº¯n

def chunk_content(text: str) -> list[str]:
    words = word_tokenize(text)
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i:i + CHUNK_SIZE]
        if len(chunk_words) < MIN_WORDS:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks

records = []
for _, row in tqdm(df_news.iterrows(), total=len(df_news)):
    for chunk in chunk_content(row["content"]):
        records.append({
            "chunk": chunk,
            "url":   row["url"],
            "date":  row["date"]
        })

df_chunks = pd.DataFrame(records)
print(f"âœ… Tá»•ng sá»‘ chunk: {len(df_chunks)}")
display(df_chunks.head(3))



# Cell 10 â€“ ghi ra file parquet Ä‘á»ƒ tÃ¡i sá»­ dá»¥ng nhanh á»Ÿ bÆ°á»›c tiáº¿p
import pathlib

WORK_DIR = pathlib.Path("/kaggle/working")
clean_path  = WORK_DIR / "news_clean.parquet"
chunk_path  = WORK_DIR / "news_chunks.parquet"

df_news.to_parquet(clean_path,  index=False)
df_chunks.to_parquet(chunk_path, index=False)

print("âœ… Ä�Ã£ lÆ°u:")
print(" â€¢", clean_path)
print(" â€¢", chunk_path)



# Cell 11 (sá»­a) â€“ cÃ i thÆ° viá»‡n, import torch, náº¡p dá»¯ liá»‡u chunk vÃ  load model embedding
!pip install -qU sentence-transformers faiss-cpu  # cháº¡y má»™t láº§n; náº¿u Ä‘Ã£ cÃ i rá»“i sáº½ bá»� qua nhanh

import pathlib, pandas as pd, numpy as np, torch
from sentence_transformers import SentenceTransformer

WORK_DIR   = pathlib.Path("/kaggle/working")
chunk_path = WORK_DIR / "news_chunks.parquet"
assert chunk_path.exists(), "KhÃ´ng tÃ¬m tháº¥y news_chunks.parquet â€“ hÃ£y cháº¯c cháº¯n Ä‘Ã£ cháº¡y Cell 10"

df_chunks = pd.read_parquet(chunk_path)
print("ğŸ”¹ Sá»‘ chunk náº¡p:", len(df_chunks))

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(EMB_MODEL_NAME, device=device)

print(f"âœ… Model loaded on {device}: {EMB_MODEL_NAME}")



# Cell 12 â€“ encode táº¥t cáº£ chunk thÃ nh vector vÃ  lÆ°u thÃ nh tá»‡p .npy
import numpy as np
from tqdm.auto import tqdm

BATCH_SIZE = 512          # Ä‘iá»�u chá»‰nh náº¿u GPU RAM háº¡n háº¹p
EMB_DIM    = embedder.get_sentence_embedding_dimension()

all_embeddings = np.empty((len(df_chunks), EMB_DIM), dtype="float32")

for start in tqdm(range(0, len(df_chunks), BATCH_SIZE)):
    end   = min(start + BATCH_SIZE, len(df_chunks))
    batch = df_chunks["chunk"].iloc[start:end].tolist()
    all_embeddings[start:end] = embedder.encode(
        batch,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,     # cosine-sim dÃ¹ng norm=1
        convert_to_numpy=True
    )

# LÆ°u xuá»‘ng WORK_DIR Ä‘á»ƒ tÃ¡i sá»­ dá»¥ng
emb_path = WORK_DIR / "news_emb.npy"
np.save(emb_path, all_embeddings)

print("âœ… Ä�Ã£ tÃ­nh xong embedding:")
print(" â€¢ Shape:", all_embeddings.shape)
print(" â€¢ LÆ°u táº¡i:", emb_path)


