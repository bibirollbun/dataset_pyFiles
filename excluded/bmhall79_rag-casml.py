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


!pip install -q transformers sentence-transformers pinecone-client datasets torch

# Confirm successful installation
print('Libraries installed successfully.')


import transformers
import sentence_transformers
import pinecone
import datasets
import torch

print('All libraries successfully imported.')


from pinecone import Pinecone, ServerlessSpec

PINECONE_API_KEY = "your-key"

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-textbook-index"

if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index_info = pc.describe_index(index_name)
host = index_info.host

index = pc.Index(index_name, host=host)

print(f'Connected to Pinecone index: {index_name}')


queries_file = "/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/queries.json"

import json
with open(queries_file, "r") as f:
    queries = json.load(f)

#display sample of queries to inspect
print("Sample queries:")
print(json.dumps(queries[:2], indent=4)) # only displays first two queries


!pip install -q pdfplumber
import pdfplumber

pdf_file = "/kaggle/input/casml-generative-ai-hackathon/Dataset_RAG (1)/book.pdf"

book_content = []
with pdfplumber.open(pdf_file) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        book_content.append({"page": i+1, "text": text})

print("Sample book content:")
print(book_content[:3]) #displays first three pages
#opting for pfdplumber > PyPDF because it was more reliable in processing the text


#chunk params
chunk_size = 512 #max num tokens per chunk
overlap = 50 #num overlap tokens between chunks

def chunk_text(text, chunk_size, overlap):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = []
for page in book_content:
    page_text = page.get("text", "")
    page_number = page.get("page", "N/A")
    if page_text.strip(): #skips empty pages
        page_chunks = chunk_text(page_text, chunk_size, overlap)
        for chunk in page_chunks:
            chunks.append({"text": chunk, "metadata": {"page": page_number}})

print("Sample chunks:")
for chunk in chunks[:2]:
    print(chunk)


import openai

openai.api_key = "open-ai-key"

def get_openai_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

for i, chunk in enumerate(chunks):
    chunk_text = chunk["text"]
    chunk_metadata = chunk["metadata"]

    embedding = get_openai_embedding(chunk_text)

    chunk_metadata["text"] = chunk_text

    index.upsert([
        {
            "id": f"chunk-{i+1}",
            "values": embedding,
            "metadata": chunk_metadata
        }
    ])

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(chunks)} chunks...")

print("All chunks have been embedded and properly stored in Pinecone.")



def retrieve_context(query, top_k=30, min_score=0.5):
    query_embedding = get_openai_embedding(query)

    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, include_values=False)

    # extract retrieved text and scores
    retrieved_chunks = []
    for match in results.get('matches', []):
        score = match.get('score', 0)
        metadata = match.get('metadata', {})

        # only keep results that have a score higher than `min_score`
        if score >= min_score and 'text' in metadata:
            retrieved_chunks.append({
                "text": metadata["text"],
                "page": metadata.get("page", "N/A"),
                "score": score
            })

    return retrieved_chunks

#helper method to reduce context for shorter queries (like definitions)
def adaptive_retrieve_context(query):
    short_query = len(query.split()) <= 5
    top_k = 10 if short_query else 30

    return retrieve_context(query, top_k=top_k)

def rerank_retrieved_context(retrieved_chunks, query):
    """
    Reorders retrieved chunks by:
        Similarity Score (higher = better)
        Keyword Overlap with Query
    """
    query_terms = set(query.lower().split())

    for chunk in retrieved_chunks:
        text_terms = set(chunk["text"].lower().split())
        chunk["keyword_overlap"] = len(query_terms & text_terms)  # Count shared words

    # Sort by highest similarity score + keyword overlap
    retrieved_chunks.sort(key=lambda x: (x["score"], x["keyword_overlap"]), reverse=True)

    return retrieved_chunks[:10]  # Keep only top-ranked results

sample_query = "What is the scientific method in psychology?"
retrieved_context = retrieve_context(sample_query)

# display retrieved results with scores
print("Retrieved Context (after increasing top_k and lowering similarity threshold):")
for i, context in enumerate(retrieved_context):
    print(f"{i+1}. Page: {context['page']} | Score: {context['score']:.4f} | Text: {context['text'][:200]}...")

# print highest and lowest scores
scores = [c["score"] for c in retrieved_context]
if scores:
    print(f"Highest Score: {max(scores):.4f} | Lowest Score: {min(scores):.4f}")
else:
    print("Still no results retrieved! Pinecone query settings might need more tuning.")



from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
qa_model = pipeline("text2text-generation", model="google/flan-t5-large", device=0)

def generate_answer(query, retrieved_chunks, max_input_tokens=400):
    context_text = " ".join([chunk["text"] for chunk in retrieved_chunks])
    tokenized_context = tokenizer.encode(context_text, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    trimmed_context = tokenizer.decode(tokenized_context[0], skip_special_tokens=True)
    prompt = (
        "Using the textbook excerpts below, provide a complete and accurate answer to the question. "
        "Ensure your response is well-structured and relevant to the text. "
        "If the textbook does not contain enough information, say so.\n\n"
        f"Textbook Excerpts:\n{trimmed_context}\n\n"
        f"Question: {query}\n\n"
        "Answer in 3-4 sentences:"
    )

    response = qa_model(prompt, max_length=100, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"]
    
    return response

sample_query = "What is the scientific method in psychology?"
generated_answer = generate_answer(sample_query, retrieved_context)

print("Generated Answer:")
print(generated_answer)


from tqdm import tqdm

batch_inputs = []
query_ids = []
retrieved_contexts = []

for query in tqdm(queries, desc="ğŸ”� Retrieving Contexts"):
    query_id = query["query_id"]
    question = query["question"]

    retrieved_context = adaptive_retrieve_context(question)
    retrieved_context = rerank_retrieved_context(retrieved_context, question)

    referenced_pages = list(set(str(chunk["page"]) for chunk in retrieved_context if "page" in chunk))
    references_json = json.dumps({"sections": [], "pages": referenced_pages})  

    context_text = " ".join([chunk["text"] for chunk in retrieved_context])
    
    tokenized_context = tokenizer.encode(context_text, truncation=True, max_length=400, return_tensors="pt") #modify max_length based on tokens allotted
    trimmed_context = tokenizer.decode(tokenized_context[0], skip_special_tokens=True)

    prompt = (
        "Using the textbook excerpts below, provide a complete and accurate answer to the question. "
        "Ensure your response is well-structured and relevant to the text. "
        "If the textbook does not contain enough information, say so.\n\n"
        f"Textbook Excerpts:\n{trimmed_context}\n\n"
        f"Question: {question}\n\n"
        "Answer in 3-4 sentences:"
    )

    # store inputs for batch processing
    batch_inputs.append(prompt)
    query_ids.append(query_id)
    retrieved_contexts.append((context_text, references_json))

print("Generating Answers in through batch...")
batch_answers = qa_model(batch_inputs, max_length=100, do_sample=True, temperature=0.7, top_p=0.9)

# convert results to submission format
submission_data = []
for i in range(len(query_ids)):
    submission_data.append({
        "ID": query_ids[i],
        "context": retrieved_contexts[i][0],
        "answer": batch_answers[i]["generated_text"],
        "references": retrieved_contexts[i][1]
    })

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as `submission.csv`.")

