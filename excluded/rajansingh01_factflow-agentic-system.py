# --------------------------
# 1. Install dependencies
# --------------------------
!pip install -q faiss-cpu sentence-transformers transformers protobuf==3.20.3

# --------------------------
# 2. Environment setup
# --------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

# --------------------------
# 3. Imports (after install)
# --------------------------
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

print("Setup complete.")



import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Small starter corpus (we will expand later)
CORPUS = [
    {
        "id": "doc1",
        "title": "Photosynthesis",
        "text": "Photosynthesis converts light energy into chemical energy in plants.",
        "source": "wiki_photosynthesis"
    },
    {
        "id": "doc2",
        "title": "CPU vs GPU",
        "text": "GPU is optimized for parallel workloads like machine learning, while CPU is designed for general tasks.",
        "source": "wiki_hardware"
    },
    {
        "id": "doc3",
        "title": "HTTP Status Codes",
        "text": "HTTP 404 means the resource was not found. HTTP 200 means the request succeeded.",
        "source": "rfc_7231"
    },
    {
        "id": "doc4",
        "title": "Machine Learning",
        "text": "Machine learning allows systems to learn patterns from data without being explicitly programmed.",
        "source": "ml_overview"
    }
]

len(CORPUS), CORPUS[:2]   # sanity check



# Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

docs = [d["text"] for d in CORPUS]
embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)

embeddings.shape



import faiss

dim = embeddings.shape[1]  # embedding dimension (should be 384)

# Create FAISS index
index = faiss.IndexFlatIP(dim)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Add embeddings to index
index.add(embeddings)

print("Index size:", index.ntotal)



def retrieve(query, top_k=3):
    # Encode query
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    
    # Search top_k documents
    D, I = index.search(q_emb, top_k)
    
    results = []
    for idx in I[0]:
        results.append(CORPUS[idx])
    return results

# Test
retrieve("What is photosynthesis?", top_k=2)



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

GEN_MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=gen_model,
    tokenizer=tokenizer,
    max_length=256,
    device=-1   # CPU; GPU is optional
)



def researcher_agent(query: str, top_k=3):
    # Step 1: Retrieve documents using your retriever
    retrieved_docs = retrieve(query, top_k=top_k)
    
    # Step 2: Convert docs into a context string
    context_block = "\n\n".join([
        f"{doc['title']}: {doc['text']} (source: {doc['source']})"
        for doc in retrieved_docs
    ])
    
    # Step 3: Draft answer prompt
    prompt = f"""
Use the following context to answer the question.

Context:
{context_block}

Question: {query}

Write a concise answer and include short inline citations such as [source].
"""
    
    # Step 4: Generate draft answer
    draft = generator(prompt, do_sample=False)[0]["generated_text"]
    
    return {
        "draft": draft,
        "retrieved_docs": retrieved_docs
    }



result = researcher_agent("What is photosynthesis?")
result



def refiner_agent(draft: str, query: str, context_docs, max_iters=2):
    """
    More robust Refiner Agent: handles empty outputs, avoids long prompts,
    and ensures revised answer is produced.
    """
    current = draft
    last_critique = ""

    # Shorten context to avoid overwhelming FLAN-T5
    context_text = "\n".join([doc["text"] for doc in context_docs][:2])

    for i in range(max_iters):
        prompt = f"""
You are a Refiner Agent. Improve the draft answer.

User Question: {query}

Relevant Context:
{context_text}

Draft Answer:
{current}

Tasks:
1. Provide a short critique (1-3 sentences).
2. Then provide an improved answer labeled EXACTLY as:

Revised:
<your improved answer>

Now provide the output:
"""
        # Generate response
        response = generator(prompt, do_sample=False, max_length=200)[0]["generated_text"]

        # Safety: if model returns nothing, stop gracefully
        if not response.strip():
            break

        # Try to extract the revised part properly
        if "Revised:" in response:
            parts = response.split("Revised:", 1)
            critique = parts[0].strip()
            revised = parts[1].strip()
        else:
            # fallback: treat everything as "revised"
            critique = "Model did not provide a critique separately."
            revised = response.strip()

        # Track and update only if revision is meaningful
        if revised and revised != current:
            current = revised
            last_critique = critique
        else:
            break

    # Final fallback: if still empty, return draft
    if not current.strip():
        current = draft

    return {
        "final_answer": current,
        "critique": last_critique
    }



test = researcher_agent("Explain photosynthesis.")
refined = refiner_agent(
    draft=test["draft"],
    query="Explain photosynthesis.",
    context_docs=test["retrieved_docs"]
)
refined



import pandas as pd

SME_DB = pd.DataFrame([
    {
        "id": "sme1",
        "name": "Dr. A. Sharma",
        "domains": "medical,health,clinical,doctor,emergency,disease,pain,chest,breathing,respiratory,care,biomedical",
        "email": "sharma@example.com",
        "rating": 4.9
    },
    {
        "id": "sme2",
        "name": "S. Rao",
        "domains": "computers,ai,ml,software,hardware,technology,deep learning,neural network",
        "email": "rao@example.com",
        "rating": 4.8
    },
    {
        "id": "sme3",
        "name": "K. Verma",
        "domains": "law,legal,policy,regulations,compliance,contract",
        "email": "verma@example.com",
        "rating": 4.7
    }
])



def is_sensitive(query: str) -> bool:
    # Medical / emergency keywords
    medical_keywords = [
        "chest pain", "difficulty breathing", "heart", "stroke", "attack",
        "emergency", "urgent", "doctor", "hospital", "medicine",
        "symptom", "sick", "tumor", "infection", "pain"
    ]
    
    # Legal keywords
    legal_keywords = [
        "illegal", "law", "policy", "court", "sue", "legal", "crime"
    ]
    
    # Financial keywords
    financial_keywords = [
        "invest", "loan", "money", "profit", "bank", "tax", "crypto"
    ]
    
    all_keywords = medical_keywords + legal_keywords + financial_keywords
    q = query.lower()
    
    return any(word in q for word in all_keywords)



def expert_matcher(query: str, top_k=1):
    q = query.lower()
    candidates = []

    # Direct keyword/domain match
    for _, row in SME_DB.iterrows():
        domain_tags = row["domains"].split(",")
        for tag in domain_tags:
            tag = tag.strip().lower()
            if tag in q:
                candidates.append(row.to_dict())
                break
    
    if candidates:
        return candidates[:top_k]

    # Semantic fallback using embeddings
    domain_texts = SME_DB["domains"].tolist()
    domain_embeddings = embedder.encode(domain_texts, convert_to_numpy=True)
    q_emb = embedder.encode([query], convert_to_numpy=True)

    sims = (domain_embeddings @ q_emb.T).flatten()
    best_idx = sims.argmax()

    return [SME_DB.iloc[best_idx].to_dict()]



expert_matcher("I have symptoms of fever and throat pain")



expert_matcher("How to optimize a neural network?")



def factflow_pipeline(query: str):
    
    # 1. Sensitivity check
    sensitive = is_sensitive(query)
    sme_match = None
    if sensitive:
        sme_match = expert_matcher(query)

    # 2. Researcher Agent â†’ draft answer
    researcher_out = researcher_agent(query)
    draft = researcher_out["draft"]
    context_docs = researcher_out["retrieved_docs"]

    # 3. Refiner Agent â†’ improved answer
    refined_out = refiner_agent(
        draft=draft,
        query=query,
        context_docs=context_docs,
        max_iters=2
    )

    final_answer = refined_out["final_answer"]
    critique = refined_out["critique"]

    # Final output object
    return {
        "query": query,
        "draft_answer": draft,
        "refined_answer": final_answer,
        "critique": critique,
        "retrieved_docs": context_docs,
        "sme_match": sme_match,
        "sensitive": sensitive
    }



result = factflow_pipeline("Explain photosynthesis.")
result



factflow_pipeline("I have chest pain and difficulty breathing, what should I do?")



demo_queries = [
    "Explain photosynthesis.",
    "What is machine learning?",
    "Difference between CPU and GPU?",
    "I have chest pain and difficulty breathing, what should I do?",
    "Is it legal to record a phone call in India?",
    "How does HTTP 404 work?",
    "What is chemical energy?",
    "My father has high fever and chest tightness.",
    "How do neural networks learn?",
    "Define parallel computing.",
    "I am thinking about investing in crypto, is it safe?",
    "How do plants convert sunlight into energy?"
]



def run_demo_evaluation(queries):
    results = []
    for i, q in enumerate(queries, start=1):
        print("\n===============================")
        print(f"QUERY {i}: {q}")
        print("===============================")
        out = factflow_pipeline(q)
        print("Draft Answer:", out["draft_answer"])
        print("Refined Answer:", out["refined_answer"])
        print("Sensitive:", out["sensitive"])
        print("SME Match:", out["sme_match"])
        print("Retrieved Docs:", [d["id"] for d in out["retrieved_docs"]])
        results.append(out)
    return results



demo_results = run_demo_evaluation(demo_queries)



CORPUS = [
    {"id": "bio1", "title": "Photosynthesis", "text": "Photosynthesis is a process in which green plants convert sunlight into chemical energy stored in glucose.", "source": "biology_basic"},
    {"id": "bio2", "title": "Cell Theory", "text": "Cell theory states that all living organisms are composed of one or more cells, and the cell is the basic structural unit of life.", "source": "biology_cell"},
    {"id": "chem1", "title": "Chemical Energy", "text": "Chemical energy is the potential of a chemical substance to undergo a chemical reaction to transform into other substances.", "source": "chemistry_basic"},
    {"id": "phys1", "title": "Gravity", "text": "Gravity is a force that attracts two bodies toward each other. On Earth, it gives weight to physical objects.", "source": "physics_gravity"},
    {"id": "geo1", "title": "Water Cycle", "text": "The water cycle describes how water evaporates, forms clouds, and returns to Earth's surface as precipitation.", "source": "geography_cycle"},
    {"id": "env1", "title": "Greenhouse Effect", "text": "The greenhouse effect occurs when gases in Earth's atmosphere trap heat, causing the planet's temperature to rise.", "source": "environment_climate"},
    {"id": "cs1", "title": "Machine Learning", "text": "Machine learning enables computers to learn patterns from data and make decisions without being explicitly programmed.", "source": "cs_ml"},
    {"id": "cs2", "title": "Neural Networks", "text": "Neural networks are computing systems inspired by biological neural structures, used for pattern recognition and prediction.", "source": "cs_nn"},
    {"id": "cs3", "title": "CPU vs GPU", "text": "A CPU handles general-purpose tasks, while a GPU is optimized for parallel processing workloads like deep learning.", "source": "cs_hardware"},
    {"id": "web1", "title": "HTTP Codes", "text": "HTTP status codes indicate the result of web requests. 200 means OK, 404 means Not Found, and 500 means Server Error.", "source": "web_http"},
    {"id": "med1", "title": "Fever Symptoms", "text": "Fever is a temporary rise in body temperature, often due to infection. Severe symptoms may require medical evaluation.", "source": "medical_fever"},
    {"id": "med2", "title": "Chest Pain Warning", "text": "Chest pain and difficulty breathing can indicate a serious medical emergency such as a heart attack and require immediate care.", "source": "medical_emergency"},
    {"id": "fin1", "title": "Inflation", "text": "Inflation is the rate at which the general level of prices for goods and services rises, reducing purchasing power.", "source": "finance_basic"},
    {"id": "fin2", "title": "Investing Risks", "text": "Investing always carries risk. High-return investments like crypto have high volatility and potential losses.", "source": "finance_risk"},
    {"id": "law1", "title": "Contract Law", "text": "Contract law governs agreements made between parties and determines when they are legally enforceable.", "source": "law_contract"},
    {"id": "law2", "title": "Privacy Law", "text": "Privacy laws regulate how personal data is collected, stored, and shared by individuals and organizations.", "source": "law_privacy"},
    {"id": "math1", "title": "Pythagorean Theorem", "text": "The Pythagorean theorem states that in a right triangle, a^2 + b^2 = c^2.", "source": "math_geometry"},
    {"id": "math2", "title": "Prime Numbers", "text": "Prime numbers are natural numbers greater than 1 that have no divisors other than 1 and themselves.", "source": "math_numbers"},
    {"id": "psy1", "title": "Cognitive Bias", "text": "Cognitive biases are systematic errors in thinking that affect decisions and judgments.", "source": "psychology_basic"},
    {"id": "his1", "title": "World War II", "text": "World War II was a global conflict lasting from 1939 to 1945 that involved most of the world's nations.", "source": "history_ww2"}
]

len(CORPUS)



docs = [d["text"] for d in CORPUS]
embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)

import faiss
dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
faiss.normalize_L2(embeddings)
index.add(embeddings)

print("New index size:", index.ntotal)



retrieve("What is photosynthesis?")



retrieve("What is chest pain?")



factflow_pipeline("I have chest pain and difficulty breathing.")



def format_final_output(result):
    answer = result["refined_answer"]
    sme = result["sme_match"]
    sensitive = result["sensitive"]

    # Base formatted output
    formatted = f"### Final Answer:\n{answer}\n\n"

    # Add SME info if sensitive
    if sensitive and sme:
        sme_data = sme[0]
        formatted += (
            "### Recommended Subject Matter Expert:\n"
            f"- **Name:** {sme_data['name']}\n"
            f"- **Specialization:** {sme_data['domains']}\n"
            f"- **Contact:** {sme_data['email']}\n"
            f"- **Rating:** {sme_data['rating']}\n\n"
        )

        # Add a safety disclaimer
        formatted += (
            "âš ï¸� **Disclaimer:** This system is not a medical or legal professional. "
            "For urgent concerns, contact a qualified expert immediately.\n\n"
        )

    return formatted



def run_factflow(query):
    raw = factflow_pipeline(query)
    return format_final_output(raw)



print(run_factflow("I have chest pain and difficulty breathing."))




with open("submission.txt", "w") as f:
    f.write("FactFlow Project Submission - File Generated Successfully.")

print("submission.txt created!")


