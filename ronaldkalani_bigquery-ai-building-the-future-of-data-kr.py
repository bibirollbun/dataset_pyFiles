from google.cloud import bigquery
import pandas as pd, json, re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import numpy as np

PROJECT_ID = "kaggle-bigquery-468910"   # <- your project
LOCATION   = "US"
client = bigquery.Client(project=PROJECT_ID, location=LOCATION)




# === Step 1 (robust): load Stack Overflow Python QA in ANY common format ===
from pathlib import Path
import json, re, pandas as pd
from typing import Iterable

BASE = Path("/kaggle/input/stack-overflow-python-qa")
assert BASE.exists(), "Attach the dataset 'stack-overflow-python-qa' to the notebook."

# Find candidate files (json, jsonl, csv, tsv, parquet)
cands = []
for pat in ("*.json", "*.jsonl", "**/*.json", "**/*.jsonl",
            "*.csv", "**/*.csv", "*.tsv", "**/*.tsv",
            "*.parquet", "**/*.parquet"):
    cands += list(BASE.glob(pat))

assert cands, f"No supported files found under {BASE}. List the folder to check what's inside."

def to_rows_pdf(df: pd.DataFrame) -> list[tuple]:
    rows = []
    # Try to map likely column names
    title_cols = ["title","question_title","Title","subject"]
    text_cols  = ["body","question_body","text","Body","Question","content","question"]
    ans_cols   = ["accepted_answer","acceptedAnswer","answer","Answer"]
    tags_cols  = ["tags","Tags","tag","categories"]

    for i, row in df.fillna("").iterrows():
        asdict = row.to_dict()

        def pick(cols: Iterable[str]) -> str:
            for c in cols:
                if c in asdict and isinstance(asdict[c], str) and asdict[c].strip():
                    return asdict[c]
            return ""

        title = pick(title_cols)
        text  = pick(text_cols)
        if not text:
            # Fallback: join all string-like columns
            text = " ".join([str(v) for v in asdict.values() if isinstance(v, str)])

        ans   = pick(ans_cols)
        tagsv = pick(tags_cols)
        # normalize tags to comma-separated
        if isinstance(tagsv, list):
            tagsv = ",".join(tagsv)
        tagsv = str(tagsv)

        # Require some letters so we keep informative docs
        if len(re.sub(r"[^A-Za-z]", "", title + " " + text)) >= 30:
            doc_id = str(asdict.get("question_id") or asdict.get("Id") or asdict.get("id") or f"r{i}")
            rows.append((doc_id, title, text if title else text, ans, tagsv))
    return rows

def load_any(fp: Path) -> list[tuple]:
    try:
        if fp.suffix.lower() == ".parquet":
            return to_rows_pdf(pd.read_parquet(fp))
        if fp.suffix.lower() in {".csv", ".tsv"}:
            sep = "\t" if fp.suffix.lower() == ".tsv" else ","
            return to_rows_pdf(pd.read_csv(fp, dtype=str, sep=sep))
        if fp.suffix.lower() in {".json", ".jsonl"}:
            # try JSON lines first
            try:
                df = pd.read_json(fp, lines=True)
                return to_rows_pdf(df)
            except Exception:
                data = json.loads(fp.read_text(errors="ignore"))
                if isinstance(data, dict): data = [data]
                return to_rows_pdf(pd.json_normalize(data))
    except Exception as e:
        print(f"[skip] {fp.name}: {e}")
    return []

all_rows = []
for fp in cands:
    rows = load_any(fp)
    if rows:
        print(f"Loaded {len(rows):,} rows from {fp.name}")
        all_rows += rows

assert all_rows, "Parsed 0 usable Q/A rows. Open a file in the sidebar to see its columns and tweak the column lists above."

df = pd.DataFrame(all_rows, columns=["doc_id","title","text","accepted_answer","tags"]).drop_duplicates("doc_id")
print("Parsed docs:", len(df))
df.head(3)



# Build TF-IDF (char n-grams) + SVD embeddings, then L2-normalize
import numpy as np, pandas as pd, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

assert {"doc_id","title","text"}.issubset(df.columns), "df must have doc_id,title,text"

texts = (df["title"].fillna("") + " " + df["text"].fillna("") + " " + df["accepted_answer"].fillna("")).tolist()

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=4096, min_df=1)
X   = vec.fit_transform(texts)

k = max(2, min(128, X.shape[0]-1, X.shape[1]-1)) if min(X.shape) > 1 else 2
Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(X) if k >= 2 else X.toarray()
Z = normalize(Z)

df_emb = df[["doc_id","title","text"]].copy()
df_emb["embedding"] = [row.astype(float).tolist() for row in Z]
print("Docs:", len(df_emb), "| Emb dim:", len(df_emb["embedding"].iloc[0]))
df_emb.head(2)



from google.cloud import bigquery

PROJECT_ID = "kaggle-bigquery-468910"   # <-- your GCP project
LOCATION   = "US"
client     = bigquery.Client(project=PROJECT_ID, location=LOCATION)

ds_id  = f"{PROJECT_ID}.hackathon_ds"
tbl_id = f"{ds_id}.docs"

# Create dataset if missing
try:
    client.get_dataset(ds_id)
except Exception:
    d = bigquery.Dataset(ds_id); d.location = LOCATION
    client.create_dataset(d, exists_ok=True)

schema = [
    bigquery.SchemaField("doc_id","STRING"),
    bigquery.SchemaField("title","STRING"),
    bigquery.SchemaField("text","STRING"),
    bigquery.SchemaField("embedding","FLOAT", mode="REPEATED"),
]

job = client.load_table_from_dataframe(
    df_emb, tbl_id,
    job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print("Uploaded:", tbl_id)



# Index helps on big tables; harmless to skip on small ones
idx_sql = f"""
CREATE VECTOR INDEX IF NOT EXISTS `{ds_id}.docs_idx`
ON `{tbl_id}` (embedding)
OPTIONS(distance_type='COSINE');
"""
try:
    client.query(idx_sql).result()
    print("Vector index ready (or already existed).")
except Exception as e:
    print("Index creation skipped:", e)



import pandas as pd, numpy as np

top_k = 5
sql = f"""
WITH q AS (SELECT doc_id, embedding FROM `{tbl_id}`)
SELECT
  query.doc_id        AS query_id,
  base.doc_id         AS neighbor_id,
  SUBSTR(base.title,1,160) AS neighbor_title,
  SUBSTR(base.text,1,220)  AS neighbor_snippet,
  distance
FROM VECTOR_SEARCH(
  TABLE `{tbl_id}`,
  'embedding',
  TABLE q,
  top_k => {top_k},
  distance_type => 'COSINE',
  options => '{{"use_brute_force": true}}'  -- safe for hackathon scale
)
WHERE query.doc_id != base.doc_id
ORDER BY query_id, distance ASC
"""
try:
    knn_df = client.query(sql).to_dataframe()
except Exception as e:
    # Fallback: local cosine KNN (still produces competition files)
    print("BigQuery VECTOR_SEARCH failed, doing local fallback ->", e)
    E = np.vstack(df_emb["embedding"].to_numpy())
    # cosine distance = 1 - dot on L2-normalized vectors
    sims = E @ E.T
    np.fill_diagonal(sims, -np.inf)
    rows = []
    ids  = df_emb["doc_id"].tolist()
    for i, qid in enumerate(ids):
        idx = np.argsort(-sims[i])[:top_k]
        for j in idx:
            rows.append((qid, ids[j], df_emb["title"].iloc[j][:160], df_emb["text"].iloc[j][:220], float(1 - sims[i, j])))
    knn_df = pd.DataFrame(rows, columns=["query_id","neighbor_id","neighbor_title","neighbor_snippet","distance"])

# Save
knn_path = "/kaggle/working/knn_results.csv"
knn_df.to_csv(knn_path, index=False)
print("Wrote:", knn_path, "| rows:", len(knn_df))
knn_df.head(5)



sub_df = (
    knn_df.sort_values(["query_id","distance"])
          .groupby("query_id", as_index=False)
          .first()[["query_id","neighbor_id","distance"]]
          .rename(columns={"query_id":"doc_id","neighbor_id":"top_match","distance":"match_distance"})
)

sub_path = "/kaggle/working/submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "| rows:", len(sub_df))
sub_df.head(5)


