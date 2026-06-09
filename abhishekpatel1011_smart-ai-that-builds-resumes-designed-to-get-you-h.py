


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


# Run this in Colab (or pip install locally)
!pip install -q sentence-transformers streamlit spacy sklearn PyPDF2 python-magic pdfminer.six

# Download small spaCy model for basic NER / POS
!python -m spacy download en_core_web_sm



import os, re, io, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# PDF reading
import PyPDF2

nlp = spacy.load("en_core_web_sm")



def extract_text_from_pdf(path_or_bytes) -> str:
    """
    Accepts a file path or bytes. Returns extracted text.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        reader = PyPDF2.PdfReader(io.BytesIO(path_or_bytes))
    else:
        reader = PyPDF2.PdfReader(open(path_or_bytes, "rb"))
    text = []
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return "\n".join(text)

# Example usage:
# text = extract_text_from_pdf('resume.pdf')



EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

# Minimal skill list (extend this)
COMMON_SKILLS = [
    "python", "java", "c++", "sql", "tensorflow", "pytorch", "keras",
    "nlp", "machine learning", "deep learning", "pandas", "numpy",
    "git", "docker", "linux", "rest", "flask", "django", "streamlit",
    "react", "matlab", "scikit-learn"
]

def extract_basic_info(text: str) -> Dict:
    doc = nlp(text[:5000])  # speed optimization: analyze initial chunk
    # Name: try NER PERSON on first lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    name = None
    if lines:
        # check first 5 lines for PERSON entity
        for ln in lines[:5]:
            doc_ln = nlp(ln)
            for ent in doc_ln.ents:
                if ent.label_ == "PERSON":
                    name = ent.text
                    break
            if name: break

    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)

    lower = text.lower()
    skills_found = [s for s in COMMON_SKILLS if s in lower]

    # education (simple heuristics)
    education = []
    for token in ["bachelor", "master", "phd", "b.sc", "m.sc", "btech", "b.tech", "mtech", "mba"]:
        if token in lower:
            education.append(token)

    return {
        "name": name,
        "emails": emails,
        "phones": phones,
        "skills": skills_found,
        "education": education
    }

# Example:
# info = extract_basic_info(text)
# print(info)



# Load a small, fast model suitable for colab / CPU: all-MiniLM-L6-v2
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast; good baseline
# You can swap to larger models later.

def embed_texts(texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)



job_postings = [
    {"id": "job1", "title": "Junior ML Engineer", "jd": "Looking for Python developer with experience in machine learning, pandas, scikit-learn, and tensorflow."},
    {"id": "job2", "title": "NLP Intern", "jd": "NLP internship: knowledge of transformers, PyTorch, NLP, tokenization, and research."},
    {"id": "job3", "title": "Data Analyst", "jd": "Strong SQL, Excel, pandas, data visualization experience. SQL queries and reporting."},
    {"id": "job4", "title": "Backend Developer", "jd": "Experience in Java, REST APIs, Docker, Kubernetes, microservices."},
]
jobs_df = pd.DataFrame(job_postings)
jobs_df['text'] = jobs_df['title'] + ". " + jobs_df['jd']
job_texts = jobs_df['text'].tolist()
job_embeddings = embed_texts(job_texts)



# Using scikit-learn NearestNeighbors with cosine distance
nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(job_embeddings)

def recommend_jobs_for_resume(resume_text: str, top_k=3) -> List[Dict]:
    # embed resume
    emb = embed_texts([resume_text])[0:1]  # shape (1, d)
    distances, indices = nn.kneighbors(emb, n_neighbors=top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        score = 1 - dist  # cosine -> similarity proxy
        job = jobs_df.iloc[idx].to_dict()
        job['score'] = float(score)
        # optional: compute which skills matched
        matched_skills = [s for s in COMMON_SKILLS if s in resume_text.lower()]
        job['matched_skills'] = matched_skills
        results.append(job)
    return results



sample_resume = """
SIDDHESH is a student skilled in Python, TensorFlow, pandas, and numpy.
Worked on deep learning projects and used scikit-learn for modelling.
Interested in NLP and machine learning.
Contact: siddhesh@example.com
"""

info = extract_basic_info(sample_resume)
print("Parsed info:", info)

recs = recommend_jobs_for_resume(sample_resume, top_k=3)
print("\nTop job matches:")
for r in recs:
    print(r['id'], r['title'], f"score={r['score']:.3f}", "matched_skills:", r['matched_skills'])


