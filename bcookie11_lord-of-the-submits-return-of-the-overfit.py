import os

# Silence TF/XLA/absl chatter that spams STDERR on Kaggle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 0=all,1=INFO,2=WARNING,3=ERROR
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/kaggle/working/st_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRAIN_Y_PATH: str = "/kaggle/input/make-data-count-finding-data-references/train_labels.csv"
TRAIN_DIR_PATH:  str = "/kaggle/input/make-data-count-finding-data-references/train"
META_PAPER_API = "https://api.crossref.org/works/{doi}"
DEFAULT_SOURCE_TYPE = 'Unknown'


# Data Helpers and Utilities 

import re
import io
import glob
import logging
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from pdfminer.high_level import extract_text
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any, Optional

import torch
import numpy as np
import urllib.parse as up
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("kaggle_notebook")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

def _read_file_binary(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _clean_ws(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # collapse >2 blank lines
    return text.strip()

def _pdf_to_text(path: str) -> str:
    """Extract text from PDF using pdfminer.six if available, else PyPDF2 as fallback."""
    # Try pdfminer.six (best quality)
    try:
        # Note: extract_text opens file internally; pass path.
        text = extract_text(path) or ""
        return _clean_ws(text)
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        import PyPDF2  # type: ignore
        text_chunks: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for pg in reader.pages:
                try:
                    s = pg.extract_text() or ""
                except Exception:
                    s = ""
                if s:
                    text_chunks.append(s)
        return _clean_ws("\n\n".join(text_chunks))
    except Exception:
        return ""

def _xml_to_text(path: str) -> str:
    """Parse XML with lxml if available, else ElementTree. Extracts title/abstract/body-ish text."""
    xml_bytes = _read_file_binary(path)

    # Try lxml first (best for namespaces/xpaths).
    try:
        from lxml import etree  # type: ignore
        parser = etree.XMLParser(recover=True, huge_tree=True)
        root = etree.fromstring(xml_bytes, parser=parser)

        # Common scholarly XML patterns (JATS-ish)
        texts: List[str] = []

        # title
        titles = root.xpath("//article-title|//title-group//article-title|//title")
        titles = [t.text if isinstance(t, etree._Element) else str(t) for t in titles]
        titles = [t for t in titles if t]
        if titles:
            texts.append("# " + titles[0].strip())

        # abstract
        abs_nodes = root.xpath("//abstract//p|//Abstract//p|//abstract")
        for n in abs_nodes:
            s = "".join(n.itertext()) if hasattr(n, "itertext") else str(n)
            s = s.strip()
            if s:
                texts.append(s)

        # body
        body_nodes = root.xpath("//body//p|//sec//p|//Body//p")
        for n in body_nodes:
            s = "".join(n.itertext()) if hasattr(n, "itertext") else str(n)
            s = s.strip()
            if s:
                texts.append(s)

        # fallback: all text
        if not texts:
            all_text = " ".join(root.itertext())
            texts = [all_text]

        return _clean_ws("\n\n".join(texts))

    except Exception:
        # Fallback to stdlib ElementTree
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_bytes)
        except Exception:
            return ""  # unreadable

        def itxt(el):
            try:
                return "".join(el.itertext())
            except Exception:
                return el.text or ""

        # Attempt similar sections by tag name
        parts: List[str] = []
        # naive title
        for tag in ("article-title", "title"):
            for n in root.iter(tag):
                s = (n.text or "").strip()
                if s:
                    parts.append("# " + s)

        # abstract
        for tag in ("abstract",):
            for n in root.iter(tag):
                s = itxt(n).strip()
                if s:
                    parts.append(s)

        # paragraphs
        for tag in ("p",):
            for n in root.iter(tag):
                s = itxt(n).strip()
                if s:
                    parts.append(s)

        if not parts:
            parts = [itxt(root)]

        return _clean_ws("\n\n".join([p for p in parts if p]))

@dataclass
class Author:
    family: Optional[str] = None
    given: Optional[str] = None
    literal: Optional[str] = None

@dataclass
class Issued:
    date_parts: List[List[int]] = field(default_factory=list)

@dataclass
class DoiResponse:
    type: str
    id: str
    categories: List[str]
    author: List[Author]
    issued: Issued
    abstract: str
    DOI: str
    publisher: str
    title: str
    URL: str
    copyright: str

    @staticmethod 
    def parse_response(data: Dict[str, Any]):
        authors = [Author(**a) for a in data.get("author", [])]
        issued = Issued(date_parts=data.get("issued", {}).get("date-parts", []))
        return DoiResponse(
            type=data.get("type", ""),
            id=data.get("id", ""),
            categories=data.get("categories", []),
            author=authors,
            issued=issued,
            abstract=data.get("abstract", ""),
            DOI=data.get("DOI", ""),
            publisher=data.get("publisher", ""),
            title=data.get("title", ""),
            URL=data.get("URL", ""),
            copyright=data.get("copyright", ""),
        )


    @staticmethod
    def parse_crossref(data: Dict[str, Any]):
        """Parse Crossref API `message` response into DoiResponse."""
        authors = [Author(**{k: v for k, v in a.items() if k in ("given","family","literal")}) 
                   for a in data.get("author", [])]
        issued = Issued(date_parts=data.get("issued", {}).get("date-parts", []))
        return DoiResponse(
            type=data.get("type", ""),
            id=data.get("DOI", ""),  # Crossref uses DOI as ID
            categories=data.get("subject", []),
            author=authors,
            issued=issued,
            abstract=data.get("abstract", ""),
            DOI=data.get("DOI", ""),
            publisher=data.get("publisher", ""),
            title="".join(data.get("title", [])) if isinstance(data.get("title"), list) else data.get("title", ""),
            URL=data.get("URL", ""),
            copyright=data.get("license", [{}])[0].get("URL", ""),
        )
        
@dataclass
class Article:
    article_id: str 
    text: str 
    extension: str 
    source: str = DEFAULT_SOURCE_TYPE
    dataset_id: str | None = None 
    dataset_id_cited: str | None = None
    embedding: np.ndarray | None = None
    file_path: str | Path | None = None 
    
    @staticmethod
    def fetch_meta_doi(doi_url: str) -> DoiResponse | None:
        try:
            headers = {"Accept": "application/vnd.citationstyles.csl+json"}
            r = requests.get(doi_url, headers=headers, timeout=30)
            if r.status_code == 200:
                result = r.json()
                return DoiResponse.parse_response(result)
                
        except Exception as e: 
            logger.error(e)
            return None
            
    @staticmethod
    def fetch_meta_crossref(doi: str) -> DoiResponse | None:
        """
        Fetch metadata for a DOI from the Crossref API and return a DoiResponse object.
        """
        try:
            api_url = f"https://api.crossref.org/works/{up.quote(doi)}"
            r = requests.get(api_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            data = r.json().get("message", {})
            return DoiResponse.parse_crossref(data)
        except Exception as e:
            logger.error("Crossref fetch failed for %s: %s", doi, e)
            return None


# Dataset ID / URL Cleaners & Converters 
import re 
# Test datasets that could possibly exist in the data
samples = [
    {
        "dataset_id": "https://doi.org/10.1098/rspb.2016.1151",
        "data": ["https://doi.org/10.5061/dryad.6m3n9"],
        "in_text_span": "The data we used in this publication can be accessed from Dryad at doi:10.5061/dryad.6m3n9.",
        "citation_type": "Primary",
    },
    {
        "dataset_id": "https://doi.org/10.1098/rspb.2018.1563",
        "data": ["https://doi.org/10.5061/dryad.c394c12"],
        "in_text_span": "Phenotypic data and gene sequences are available from the Dryad Digital Repository: http://dx.doi.org/10.5061/dryad.c394c12",
        "citation_type": "Primary",
    },
    {
        "dataset_id": "https://doi.org/10.1534/genetics.119.302868",
        "data": ["https://doi.org/10.25386/genetics.11365982"],
        "in_text_span": "The authors state that all data necessary for confirming the conclusions presented in the article are represented fully within the article. Supplemental material available at figshare: https://doi.org/10.25386/genetics.11365982.",
        "citation_type": "Primary",
    },
    {
        "dataset_id": "https://doi.org/10.1038/sdata.2014.33",
        "data": ["GSE37569", "GSE45042", "GSE28166"],
        "in_text_span": "Primary data for Agilent and Affymetrix microarray experiments are available at the NCBI Gene Expression Omnibus (GEO, http://www.ncbi.nlm.nih.gov/geo/) under the accession numbers GSE37569, GSE45042 , GSE28166",
        "citation_type": "Primary",
    },
    {
        "dataset_id": "https://doi.org/10.12688/wellcomeopenres.15142.1",
        "data": ["pdb 5yfp"],
        "in_text_span": "Figure 1. Evolution and structure of the exocyst... All structural images were modelled by the authors from PDB using UCSF Chimera.",
        "citation_type": "Secondary",
    },
    {
        "dataset_id": "https://doi.org/10.3389/fimmu.2021.690817",
        "data": ["E-MTAB-10217", "PRJE43395"],
        "in_text_span": "The datasets presented in this study can be found in online repositories. The names of the repository/repositories and accession number(s) can be found below: https://www.ebi.ac.uk/arrayexpress/, E-MTAB-10217 and https://www.ebi.ac.uk/ena, PRJE43395.",
        "citation_type": "Secondary",
    },
]

ACCESSION_PATTERNS = [
    # DOI (bare "10." prefix, or full http(s) doi.org link, or "doi:10...")
    (re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)?(10\.\d{4,9}/\S+)$", re.I),
     lambda m: f"https://doi.org/{m.group(1)}"),

    # GEO (Gene Expression Omnibus)
    (re.compile(r"^GSE\d+$", re.I),
     lambda m: f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={m.group(0)}"),

    # ENA run/experiment (ERR/ERS/SRR/DRR/etc.)
    (re.compile(r"^(ERR|ERS|SRR|SRX|SRP|DRR|DRX|DRP|ERX|ERP)\d+$", re.I),
     lambda m: f"https://www.ebi.ac.uk/ena/browser/view/{m.group(0)}"),

    # dbSNP rs IDs
    (re.compile(r"^rs\d+$", re.I),
     lambda m: f"https://www.ncbi.nlm.nih.gov/snp/{m.group(0)}"),

    # PDB (4-char alphanumeric IDs)
    (re.compile(r"^[0-9A-Za-z]{4}$"),
     lambda m: f"https://www.rcsb.org/structure/{m.group(0)}"),

    # ChEMBL compounds/targets/assays
    (re.compile(r"^CHEMBL\d+$", re.I),
     lambda m: f"https://www.ebi.ac.uk/chembl/compound_report_card/{m.group(0)}/"),

    # DDBJ/GenBank/RefSeq nucleotide accessions (D10700, CP013147, NC_#######)
    (re.compile(r"^(?:[A-Z]{1,2}\d{5,6}|NC_\d+)$", re.I),
     lambda m: f"https://www.ncbi.nlm.nih.gov/nuccore/{m.group(0)}"),
]

def resolve_accession(acc: str) -> Optional[str]:
    """Return a URL for any accession/identifier/DOI."""
    
    if acc is None or (isinstance(acc, float) and pd.isna(acc)):
        return None
        
    s = str(acc).strip()
    if not s:
        return None

    # Try regex patterns
    for pattern, builder in ACCESSION_PATTERNS:
        m = pattern.match(s)
        if m:
            return builder(m)

    # Special-case string prefixes
    if s.upper().startswith("ENS"):  # Ensembl
        return f"https://www.ensembl.org/id/{s}"
    if s.upper().startswith("IPR"):  # InterPro
        return f"https://www.ebi.ac.uk/interpro/entry/{s.upper()}"
    if s.upper().startswith("CVCL_"):  # Cellosaurus
        return f"https://www.cellosaurus.org/{s.upper()}"
    if s.upper().startswith("EMPIAR-"):  # EMPIAR
        return f"https://www.ebi.ac.uk/empiar/{s.upper()}"
    if s.upper().startswith("HGNC:"):  # HGNC gene IDs
        return f"https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/{s.upper()}"
    if re.match(r"^K\d{5}$", s, flags=re.I):  # KEGG Orthology
        return f"https://www.genome.jp/dbget-bin/www_bget?ko:{s.upper()}"
    if s.upper().startswith("EPI_ISL_"):  # GISAID
        return f"https://www.gisaid.org/search?query={s}"

    # If it's already an HTTP(S) URL but didn't match DOI/PDB etc., keep as-is
    if s.lower().startswith("http"):
        return s

    # Fallback
    return s


import numpy as np 
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer

tqdm.pandas()

MODEL_NAME = "all-MiniLM-L6-v2"
K = 5
# Data fields IDs
# dropped all in the list
ID_LABELS = ["dataset_id", "article_id", "id", "DOI", "url"]
# data columns included in the input batch
# dropped: issued, embedding, author, text
TRAIN_LABELS = [ 'title', 'segments', 'extension', 'abstract', 'publisher', 'copyright', 'issued_year', 'all_authors', 'categories']
ALL_FIELDS = [
    'article_id','text','extension','source','dataset_id','dataset_id_cited','type',
    'id','categories','abstract','DOI','publisher','title','URL','copyright',
    'issued_year','all_authors','y'
]

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    # Extracts the issued year from the issued date
    df["issued_year"] = (
        pd.json_normalize(df["issued"])
        .explode("date_parts")["date_parts"]
        .map(lambda x: int(x[0]) if isinstance(x, np.ndarray) and len(x) > 0 else None)
    )
    # EXTRACT AUTHORS

    author_snippets = [
        [
            " ".join(k for k in (j.get("family"), j.get("given"), j.get("literal")) if k)
            for j in i
        ]
        if isinstance(i, (list, np.ndarray))
        else tuple()
        for i in df["author"].values
    ]

    assert len(author_snippets) == df.shape[0], (
        f"Expected {df.shape[0]} number of authors (tuples) in dataset but parsed {len(author_snippets)}. Fix query."
    )

    df['all_authors'] = author_snippets
    output_fields = [i for i in ALL_FIELDS if i in df.columns]
    
    # CREATE TARGET LABELS
    if 'source' in df.columns:
        df["y"] = df["source"].map({"Primary": 0, "Secondary": 1, "Missing": 2, "Unknown": 3})
        return df[~df['source'].isin(['Missing', 'Unknown'])][output_fields]
        
    return df[output_fields]
    
def load_train_dataset():
    """ Loads the dataset for training. The train data has labels from `../train_labels.csv`. Label fields are: (`article_id`, `dataset_id`, `type`) i.e. the predicting variables. """
    
    targets = pd.read_csv(TRAIN_Y_PATH)
    logger.info(f"Total distinct ref type Labels: {targets['type'].unique()}")
    
    for path in tqdm(Path("/kaggle/input").rglob("*"), desc="Loaded Train dataset."):
        if path.parents[1].stem == 'train' and path.is_file():
            info = {}
            
            ext = path.suffix
            
            if ext == '.pdf': 
                text = _pdf_to_text(str(path))
            elif ext == '.xml': 
                text = _xml_to_text(str(path))

            meta = targets[targets['article_id'] == path.stem]
            
            info['extension'] = ext 
            info['text'] = text 
            info['article_id'] = path.stem
            info['file_path'] = str(path)
            # info['embedding'] = model.encoder(info['text']) if text != '' else None
            
            if not meta.empty:
                # only gets the first data entry meta . . .
                metas = meta.iloc[0].to_dict()
                info["source"] = metas.get("type", DEFAULT_SOURCE_TYPE)
                info["dataset_id"] = metas.get("dataset_id", None)
                info["dataset_id_cited"] = resolve_accession(info["dataset_id"]) if info["dataset_id"] is not None else None
            else:
                logger.warning("No metadata found for %s", path.stem)

            yield Article(**info)


# Main Caller 

class DoiData(Dataset): 
    """ Doi Dataset handler. 
    Target types:
    'Unknown': missing from train dataset
    'Missing': missing from data - predefined in the dataset
    'Primary' / 'Secondary': Main Data Referencing Labels
    """
    
    def __init__(self):
        self.data = list(load_train_dataset())

        assert len(self.data) > 0, "Empty dataset loaded to instance. Pls reconfigure path or data parser."
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int): 
        if idx > len(self.data) or idx < 0:
            raise ValueError('Index out of range. Pls reconfigure processor.')

        # TODO: CONVERT TO X_train, y_train outputs
        article = self.data[idx]
        
        if article.dataset_id_cited:
            if meta := Article.fetch_meta_doi(article.dataset_id_cited):
                return {**asdict(article), **asdict(meta)}
                
        return asdict(article)

def setup_test_dataset() -> pd.DataFrame:
    parquet_path = Path("/kaggle/working/test_dataset.parquet")

    # If parquet already exists, load and return it
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    # Otherwise build from scratch
    test_data = []
    
    for path in tqdm(Path("/kaggle/input").rglob("*"), desc="Loaded Test dataset"):
        if path.is_file() and path.parents[1].stem == "test":
            info = {}
            
            file_path = str(path)
            ext = path.suffix
            stem = path.stem
            doi = stem.replace("_", "/", 1)
            
            print(f"Retrieved test: {doi}")
            
            if ext == ".xml":
                text = _xml_to_text(file_path)
            elif ext == ".pdf":
                text = _pdf_to_text(file_path)
            else:
                continue  # skip unsupported files
            
            info["extension"] = ext
            info["text"] = text
            info["article_id"] = path.stem
            info["file_path"] = file_path
            info["dataset_id"] = resolve_accession(doi)
            info["dataset_id_cited"] = resolve_accession(doi)
            
            if meta := Article.fetch_meta_crossref(info["dataset_id_cited"]):
                info.update(**asdict(meta))
            
            test_data.append(info)
    
    test_data = pd.DataFrame.from_records(test_data)
    test_dataset = prepare_data(test_data)

    # Save for reuse
    test_dataset.to_parquet(parquet_path)
    # test_dataset.to_csv(parquet_path.with_suffix(".csv"), index=False)

    return test_dataset
    
def setup_train_dataset() -> pd.DataFrame:
    parquet_path = Path("/kaggle/working/train_dataset.parquet")

    # If parquet already exists, load and return it
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    # Otherwise build from scratch
    ds = DoiData()
    full_data = list(ds)
    data = pd.DataFrame.from_records(full_data)
    train_dataset = prepare_data(data)

    # Save for reuse
    train_dataset.to_parquet(parquet_path)
    # train_dataset.to_csv(parquet_path.with_suffix(".csv"), index=False)

    return train_dataset


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

tqdm.pandas()

MODEL_NAME = "all-MiniLM-L6-v2"
K = 5
ID_LABELS = ["dataset_id", "article_id", "id", "DOI", "url"]
TRAIN_LABELS = [
    'title', 'segments', 'extension', 'abstract', 'publisher',
    'copyright', 'issued_year', 'all_authors', 'categories'
]
ALL_FIELDS = [
    'article_id','text','extension','source','dataset_id','dataset_id_cited','type',
    'id','categories','abstract','DOI','publisher','title','URL','copyright',
    'issued_year','all_authors','y'
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Basement(SentenceTransformer):
    def __init__(self, initial_data: pd.DataFrame):
        super().__init__(MODEL_NAME, device=DEVICE)
        self.x = self.preprocess_data(initial_data)
        self.to(DEVICE)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def _meta_conditioner(self, row):
        combined_prompt = f"""
        Title: {row.get('title', '')}
        Abstract: {row.get('abstract', '')}
        Author: {row.get('all_authors', '')}
        Extension File Type: {row.get('extension', '')}
        Publisher: {row.get('publisher', '')}
        Category: {row.get('categories', '')}
        Issued Year: {row.get('issued_year', '')}
        Copyright: {row.get('copyright', '')}
        """
        meta_embedding = self.encode(
            [combined_prompt],
            convert_to_numpy=True,
            output_value="sentence_embedding",
            show_progress_bar=False
        )
        text_embedding = row.get('segments', None)

        if isinstance(text_embedding, np.ndarray):
            if text_embedding.ndim == 1:
                text_embedding = np.expand_dims(text_embedding, axis=0)
            return np.vstack((meta_embedding, text_embedding))

        return meta_embedding  # fallback to meta only

    def _segment_text(self, s: str, k: int = K) -> np.ndarray:
        if not isinstance(s, str) or not s:
            return np.zeros((k, self.get_sentence_embedding_dimension()), dtype=np.float32)
        n = len(s)
        idx = np.linspace(0, n, k + 1, dtype=int)
        chunks = [s[idx[i]:idx[i+1]] for i in range(k)]
        emb = self.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32).reshape(k, -1)

    def preprocess_data(
        self,
        data: pd.DataFrame,
        k: int = K,
        train_labels: list[str] | None = None
    ) -> pd.DataFrame:
        train_labels = train_labels or TRAIN_LABELS
        df = data.copy()

        # 1) create 'segments'
        df["segments"] = df["text"].apply(lambda x: self._segment_text(x, k))

        # 2) explode multi-valued columns
        for col in ("all_authors", "segments", "categories"):
            if col in df.columns:
                df = df.explode(col, ignore_index=True)

        # 3) drop duplicated rows
        subset_cols = [c for c in train_labels if c in df.columns and c != "segments"]
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep="first")

        # 4) fills / cleaning
        if "issued_year" in df.columns:
            mode_series = df["issued_year"].dropna().mode()
            if not mode_series.empty:
                df["issued_year"] = df["issued_year"].fillna(mode_series.iloc[0])

        if "copyright" in df.columns:
            df["copyright"] = (
                df["copyright"].astype("string").str.strip().replace({"": "Unknown"})
            )

        if "categories" in df.columns:
            df["categories"] = df["categories"].astype("string").fillna("Unknown")

        df["inputs"] = df.apply(self._meta_conditioner, axis=1)
        
        return df

    def predict_from_temporal_state(self, inputs: dict, k: int = 1):
        """
        Greedy temporal approximation:
        - Approximates article_id, dataset_id, and source/type by maximizing cosine similarity step by step.
        - Similar to beam search but with greedy (top1) selection.
        """
        text = inputs.get("text")
        if not text:
            print("Skipping this dataset as it is probably hidden test set.")
            return None

        # Encode query text
        segment = self._segment_text(text)
        inputs = dict(inputs, segments=segment)
        input_emb = self._meta_conditioner(inputs)

        if input_emb.ndim == 1:
            input_emb = input_emb.reshape(1, -1)

        q = input_emb.mean(axis=0, keepdims=True)
        q_tensor = torch.tensor(q, device=DEVICE, dtype=torch.float32)

        # --- Step 1: Greedy match for article_id ---
        article_scores = {}
        for aid, row in self.x.groupby("article_id"):
            valid_inputs = [
                x for x in row["inputs"].values
                if isinstance(x, np.ndarray) and x.size > 0
            ]
            if not valid_inputs:
                continue
            train_tensor = torch.tensor(
                np.vstack(valid_inputs), device=DEVICE, dtype=torch.float32
            )
            article_scores[aid] = float(cos_sim(train_tensor, q_tensor).mean())

        if not article_scores:
            return None
        best_article = max(article_scores.items(), key=lambda x: x[1])[0]

        # --- Step 2: Greedy match for dataset_id (within chosen article) ---
        dataset_scores = {}
        for did, row in self.x[self.x["article_id"] == best_article].groupby("dataset_id"):
            valid_inputs = [
                x for x in row["inputs"].values
                if isinstance(x, np.ndarray) and x.size > 0
            ]
            if not valid_inputs:
                continue
            train_tensor = torch.tensor(
                np.vstack(valid_inputs), device=DEVICE, dtype=torch.float32
            )
            dataset_scores[did] = float(cos_sim(train_tensor, q_tensor).mean())

        if not dataset_scores:
            return {"article_id": best_article}

        best_dataset = max(dataset_scores.items(), key=lambda x: x[1])[0]

        # --- Step 3: Greedy match for source/type (within chosen dataset) ---
        type_scores = {}
        subset = self.x[
            (self.x["article_id"] == best_article) & (self.x["dataset_id"] == best_dataset)
        ]
        for source, row in subset.groupby("source"):
            valid_inputs = [
                x for x in row["inputs"].values
                if isinstance(x, np.ndarray) and x.size > 0
            ]
            if not valid_inputs:
                continue
            train_tensor = torch.tensor(
                np.vstack(valid_inputs), device=DEVICE, dtype=torch.float32
            )
            type_scores[source] = float(cos_sim(train_tensor, q_tensor).mean())

        if not type_scores:
            return {"article_id": best_article, "dataset_id": best_dataset}

        best_type, best_score = max(type_scores.items(), key=lambda x: x[1])

        return {
            "article_id": best_article,
            "dataset_id": best_dataset,
            "type": best_type,
            "score": best_score,
        }
        
    def predict(self, inputs: dict, threshold: float = 0.5):
        # disadvantage of this method is that it is strictly greater than previous scores .. can be shit 
        
        text = inputs.get("text")
        
        if not text:
            print('Skipping this dataset as it is probably hidden test set.')
            # measure against seen 
            return None
            
        # Query embedding
        segment = self._segment_text(text)
        inputs = dict(inputs, segments=segment)
        input_emb = self._meta_conditioner(inputs)
        
        if input_emb.ndim == 1:
            input_emb = input_emb.reshape(1, -1)
            
        q = input_emb.mean(axis=0, keepdims=True)

        q_tensor = torch.tensor(q, device=DEVICE, dtype=torch.float32)

        max_pred = None
        for grp, row in self.x.groupby(['article_id', 'dataset_id', 'source']):
            valid_inputs = [
                x for x in row['inputs'].values
                if isinstance(x, np.ndarray) and x.size > 0
            ]
            if not valid_inputs:
                continue

            train_sample = np.vstack(valid_inputs)
            train_tensor = torch.tensor(train_sample, device=DEVICE, dtype=torch.float32)

            avg = float(cos_sim(train_tensor, q_tensor).mean())
            prev = max_pred['score'] if max_pred else None
            if (max_pred is None) or (avg > prev):
                source_type = grp[2] if avg >= threshold else (
                    "Primary" if grp[2] == "Secondary" else "Secondary" if grp[2] == "Primary" else grp[2]
                )
                
                max_pred = {
                    'article_id': grp[0],
                    'dataset_id': grp[1],
                    'type': source_type,
                    'score': avg,
                }
        return max_pred



#test_dataset.to_parquet('/kaggle/working/test_dataset.parquet')
#train_dataset.to_parquet('/kaggle/working/train_dataset.parquet')


train_data = setup_train_dataset()
test_data = setup_test_dataset()

base = Basement(train_data)
prediction = []

for _, row in test_data.iterrows():
    pred = base.predict(row.to_dict())
    if pred:
        prediction.append(pred)

submission = pd.DataFrame.from_records(prediction)
submission = submission.sort_values(["article_id", "dataset_id", "type"]).reset_index(drop=True)
submission['row_id'] = submission.index

submission[['row_id', 'article_id', 'dataset_id', 'type']].to_csv('/kaggle/working/submission.csv', index=False)
submission[['row_id', 'article_id', 'dataset_id', 'type']].to_csv('submission.csv', index=False)
print(f'data submitted! Saving {submission.shape[0]} submissions :3', submission.head(5))


submission

