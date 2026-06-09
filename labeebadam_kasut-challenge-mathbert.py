# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Kaggle Notebook: TFâ€“IDF + Linear SVC with MathTextProcessor
# # ================================================================
# # Endâ€‘toâ€‘end pipeline that:
# #   1. Builds richlyâ€‘preprocessed text columns with MathTextProcessor
# #   2. Feeds four text views (tfidf_text, tfidf_keywords, bert_text, bert_keywords)
# #      into a ColumnTransformer of independent TFâ€“IDF vectorizers
# #   3. Gridâ€‘searches a LinearSVC classifier
# #   4. Evaluates on a stratified holdâ€‘out set and writes submission.csv

# # -----------------------------
# # 1) Setup imports
# # -----------------------------
# import warnings
# import re
# from typing import Dict, List

# import pandas as pd
# import numpy as np

# # NLTK may be missing on some Kaggle sessions â†’ install onâ€‘theâ€‘fly
# try:
#     import nltk  # noqa: F401
# except ModuleNotFoundError:
#     import subprocess, sys
#     print("Installing NLTK â€¦ (oneâ€‘off, cached for the session)")
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nltk"])
#     import nltk  # noqa: F401

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score, classification_report, f1_score

# # Download minimal NLTK corpora only if not already available
# for resource in ("punkt", "stopwords", "wordnet"):
#     try:
#         nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
#     except LookupError:
#         nltk.download(resource, quiet=True)
# # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# import warnings
# import re
# from typing import Dict, List

# import pandas as pd
# import numpy as np
# import nltk

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score, classification_report, f1_score

# # Download minimal NLTK corpora
# nltk.download("punkt", quiet=True)
# nltk.download("stopwords", quiet=True)
# nltk.download("wordnet", quiet=True)


# # Optional BERT tokenizer â”€ fall back gracefully if transformers is missing
# try:
#     from transformers import BertTokenizer
# except ModuleNotFoundError:          # library not installed
#     BertTokenizer = None             # sentinel for â€œno tokenizerâ€�
#     print("transformers not found â€“ proceeding without BERT tokenizer")

# from typing import Optional, Any   # add at the imports

# class MathTextProcessor:
#     """Generates multiple processed text columns for mathâ€‘wordâ€‘problem datasets.

#     Creates four columns:
#         * bert_text         â€“ tokens tailored for BERT with special markers
#         * bert_keywords     â€“ keywordâ€‘only view from bert_text
#         * tfidf_text        â€“ cleaned plain text for standard TFâ€“IDF
#         * tfidf_keywords    â€“ keywordâ€‘only view from tfidf_text
#     """

#     def __init__(self, bert_tokenizer: Optional[Any] = None):            
#         self.tokenizer = bert_tokenizer
#         self._init_patterns()
#         self._init_special_tokens()
#         if self.tokenizer is not None:
#             self._add_special_tokens()

#     # ---------------- Internal helpers ----------------
#     def _init_patterns(self):
#         # Anything between $...$, \(...\), \[...\] or common LaTeX environments
#         self.latex_pattern = re.compile(
#             r"\$(?:\\\$|[^\$])*?\$|\\\(.*?\\\)|\\\[.*?\\\]|"  # inline/display maths
#             r"\\begin\{equation\}.*?\\end\{equation\}|"            # equation
#             r"\\begin\{align\}.*?\\end\{align\}|"                  # align
#             r"\\[a-zA-Z]+\{.*?\}|\\[a-zA-Z]+\\?",                  # single commands
#             re.DOTALL
#         )

#         # Maps raw math symbols to textual placeholders
#         self.math_symbols_map: Dict[str, Dict[str, str]] = {
#              "bert": {
#                 r"\+": " [PLUS] ",
#                 r"-"  : " [MINUS] ",
#                 r"\*" : " [MUL] ",
#                 r"/"  : " [DIV] ",
#                 r"="  : " [EQ] ",
#                 r"\^" : " [POW] ",
#                 r"<"  : " [LT] ",
#                 r">"  : " [GT] ",
#                 r"\\leq"  : " [LEQ] ",
#                 r"\\geq"  : " [GEQ] ",
#                 r"\\times": " [MUL] ",
#                 r"\\div"  : " [DIV] ",
#                 r"\\cdot" : " [MUL] ",
#                 r"\\pm"   : " [PLUSMINUS] ",
#                 r"\\mp"   : " [MINUSPLUS] ",
#             },
#             "tfidf": {
#                 r"\+": " plus ",
#                 r"-"  : " minus ",
#                 r"\*" : " times ",
#                 r"/"  : " divided_by ",
#                 r"="  : " equals ",
#                 r"\^" : " power ",
#                 r"<"  : " less_than ",
#                 r">"  : " greater_than ",
#                 r"\\leq"  : " less_or_equal ",
#                 r"\\geq"  : " greater_or_equal ",
#                 r"\\times": " times ",
#                 r"\\div"  : " divided_by ",
#                 r"\\cdot" : " dot ",
#                 r"\\pm"   : " plus_or_minus ",
#                 r"\\mp"   : " minus_or_plus ",
#             },
#         }

#         self.number_pattern = re.compile(r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b")
#         self.whitespace_pattern = re.compile(r"\s+")

#     def _init_special_tokens(self):
#         self.special_tokens: List[str] = [
#             '[MATH]', '[PLUS]', '[MINUS]', '[MUL]', '[DIV]', '[EQ]', '[POW]',
#             '[LT]', '[GT]', '[LEQ]', '[GEQ]', '[PLUSMINUS]', '[MINUSPLUS]',
#             '[FIND]', '[NUMBER]', '[VALUE]', '[LET]', '[EQUATION]', '[POINTS]',
#             '[POSITIVE]', '[INTEGER]', '[TRIANGLE]', '[SUM]', '[SEQUENCE]',
#             '[ANGLE]', '[SQUARE]', '[DIGITS]', '[DIVISIBLE]', '[ROOTS]',
#             '[CIRCLE]', '[RADIUS]', '[PRODUCT]', '[EQUAL]', '[LINE]', '[DISTANCE]',
#             '[TIME]', '[PROBABILITY]', '[PRIME]', '[DIGIT]', '[FIRST]', '[SECOND]',
#             '[THIRD]', '[DISTINCT]', '[POINT]'
#         ]

#         self.keyword_to_token: Dict[str, str] = {
#             'find': '[FIND]',
#             'number': '[NUMBER]',
#             'value': '[VALUE]',
#             'let': '[LET]',
#             'equation': '[EQUATION]',
#             'points': '[POINTS]',
#             'positive': '[POSITIVE]',
#             'integer': '[INTEGER]',
#             'triangle': '[TRIANGLE]',
#             'sum': '[SUM]',
#             'sequence': '[SEQUENCE]',
#             'angle': '[ANGLE]',
#             'square': '[SQUARE]',
#             'digits': '[DIGITS]',
#             'digit': '[DIGIT]',
#             'divisible': '[DIVISIBLE]',
#             'roots': '[ROOTS]',
#             'circle': '[CIRCLE]',
#             'radius': '[RADIUS]',
#             'product': '[PRODUCT]',
#             'equal': '[EQUAL]',
#             'line': '[LINE]',
#             'distance': '[DISTANCE]',
#             'time': '[TIME]',
#             'probability': '[PROBABILITY]',
#             'prime': '[PRIME]',
#             'first': '[FIRST]',
#             'second': '[SECOND]',
#             'third': '[THIRD]',
#             'distinct': '[DISTINCT]',
#             'point': '[POINT]'
#         }

#     def _add_special_tokens(self):
#         """Register special math tokens into the provided tokenizer."""
#         self.tokenizer.add_tokens(self.special_tokens)

#     # -------------- Public preprocessing --------------
#     def preprocess_for_bert(self, text: str) -> str:
#         if not isinstance(text, str):
#             return ""
#         # Replace LaTeX snippets with generic marker
#         text = self.latex_pattern.sub('[MATH]', text)
#         # Substitute raw math symbols with bracketed tokens
#         for pattern, repl in self.math_symbols_map['bert'].items():
#             text = re.sub(pattern, repl, text)
#         # Replace common keywords with bracketed tokens
#         for word, token in self.keyword_to_token.items():
#             text = re.sub(rf"\b{word}\b", token, text, flags=re.IGNORECASE)
#         # Clean miscellaneous characters
#         text = text.lower()
#         text = re.sub(r"[^a-z0-9\s\[\]]", "", text)
#         text = self.whitespace_pattern.sub(' ', text).strip()
#         return text

#     def preprocess_for_tfidf(self, text: str) -> str:
#         if not isinstance(text, str):
#             return ""
#         # Show presence of LaTeX explicitly
#         text = self.latex_pattern.sub('math_expression', text)
#         for pattern, repl in self.math_symbols_map['tfidf'].items():
#             text = re.sub(pattern, repl, text)
#         text = self.number_pattern.sub('number_token', text)
#         text = text.lower()
#         text = re.sub(r"[^a-z0-9\s_]", "", text)
#         text = self.whitespace_pattern.sub(' ', text).strip()
#         return text

#     def extract_keywords(self, text: str) -> str:
#         if not isinstance(text, str):
#             return ""
#         tokens = text.lower().split()
#         keywords: List[str] = []
#         for tok in tokens:
#             if tok.startswith('[') and tok.endswith(']'):
#                 keywords.append(tok.upper())
#             elif tok in self.keyword_to_token:
#                 keywords.append(self.keyword_to_token[tok])
#         return ' '.join(keywords)

#     # ------------------- DataFrame API -------------------
#     def process_dataframe(self, df: pd.DataFrame, text_column: str = 'Question') -> pd.DataFrame:
#         df = df.copy()
#         # Build new columns
#         df['bert_text']      = df[text_column].apply(self.preprocess_for_bert)
#         df['bert_keywords']  = df['bert_text'].apply(self.extract_keywords)
#         df['tfidf_text']     = df[text_column].apply(self.preprocess_for_tfidf)
#         df['tfidf_keywords'] = df['tfidf_text'].apply(self.extract_keywords)
#         return df

# # ----------------------------------------------------
# # 3) Load raw Kaggle data
# # ----------------------------------------------------
# train = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
# test  = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')

# # ----------------------------------------------------
# # 4) Generate processed columns
# # ----------------------------------------------------
# try:
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') if BertTokenizer else None
# except Exception:
#     tokenizer = None
# processor = MathTextProcessor(tokenizer)

# train_p = processor.process_dataframe(train)
# test_p  = processor.process_dataframe(test)

# FILL = "emptytoken"          # any simple word

# for df in (train_p, test_p):
#     for col in ['tfidf_keywords', 'bert_keywords']:
#         df[col] = (
#             df[col]
#               .fillna("")                         # just in case
#               .replace(r"^\s*$", FILL, regex=True)   # â†� key line
#         )

# train_p.head(10)


# ===========================================================
# ONE-CELL NB-SVM  +  Char TF-IDF  +  RandomOverSampler SVC
# ===========================================================
import importlib, subprocess, sys, re
from pathlib import Path

# ---------- install exact versions *before* import ----------
def need(pkg, ver=""):
    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"{pkg}{ver}"])

need("imbalanced_learn", "==0.13.0")   # compatible w/ current sklearn
need("nltk")

# ---------- imports ----------
import numpy as np, pandas as pd, nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.svm import LinearSVC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

nltk.download("punkt", quiet=True)

# ---------- tiny cleaner ----------
TOKEN = r"(?u)\b\w+\b"
def clean(t:str)->str:
    t = re.sub(r"\$(?:\\\$|[^\$])*?\$", " math ", t)
    t = re.sub(r"[^A-Za-z0-9]+", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()

# ---------- NB-SVM transformer ----------
class NBSVM(BaseEstimator, TransformerMixin):
    def __init__(self, ngram=(1, 2), min_df=2, sublinear_tf=True):
        self.ngram = ngram               # â†� NEW
        self.min_df = min_df             # â†� NEW
        self.sublinear_tf = sublinear_tf # â†� NEW
        self.tfidf = TfidfVectorizer(token_pattern=TOKEN,
                                     ngram_range=ngram,
                                     min_df=min_df,
                                     sublinear_tf=sublinear_tf)
    def fit(self, X, y):
        X = self.tfidf.fit_transform(X); y = np.asarray(y)
        def lr(c):
            p = X[y==c].sum(axis=0)+1
            q = X[y!=c].sum(axis=0)+1
            return np.log(p/q)
        self.r_ = np.asarray(np.mean([lr(c) for c in np.unique(y)], 0)).ravel()
        return self
    def transform(self, X):
        return self.tfidf.transform(X).multiply(self.r_)

# ---------- data ----------
DATA = Path("/kaggle/input/classification-of-math-problems-by-kasut-academy")
train = pd.read_csv(DATA/"train.csv")
test  = pd.read_csv(DATA/"test.csv")
train["clean"] = train["Question"].apply(clean)
test ["clean"] = test ["Question"].apply(clean)

X = train[["clean"]]; y = train["label"]

# ---------- feature union ----------
feat = ColumnTransformer([
    ("nbsvm", NBSVM(), "clean"),
    ("char",  TfidfVectorizer(analyzer="char",
                              ngram_range=(3,5),
                              min_df=2,
                              sublinear_tf=True), "clean")
])

# ---------- oversampled SVC pipeline ----------
pipe = ImbPipeline([
    ("ros",  RandomOverSampler(random_state=42)),
    ("feat", feat),
    ("clf",  LinearSVC(class_weight="balanced"))
])

grid = GridSearchCV(
    pipe,
    {"clf__C":[0.5,1,2,4]},
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
).fit(X, y)

print("\nğŸŸ¢  Best CV macro-F1:", grid.best_score_, grid.best_params_)

# ---- optional hold-out sanity check ----
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123)
best = grid.best_estimator_.fit(X_tr, y_tr)
pred = best.predict(X_val)
print("\nHold-out accuracy", accuracy_score(y_val,pred),
      "F1-macro", f1_score(y_val,pred,average="macro"))
print(classification_report(y_val, pred))

# ---------- final fit & submission ----------
best.fit(X, y)
test_pred = best.predict(test[["clean"]])
pd.DataFrame({"id":test["id"], "label":test_pred}) \
  .to_csv("/kaggle/working/submission.csv", index=False)

print("\nâœ… submission.csv saved to /kaggle/working/")


