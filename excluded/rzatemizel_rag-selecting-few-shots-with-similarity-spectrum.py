%%writefile constants_rag.py

# --- Model Paths ---
# Model for retrieving similar examples (Encoder)
EMBEDDING_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"

# Model for reasoning and final classification (Decoder)
GENERATIVE_MODEL_PATH = "/kaggle/input/qwen2.5/transformers/14b-instruct-awq/1"

# --- Data Path ---
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

# --- RAG Hyperparameters ---
# --- NEW: Parameters for Similarity Spectrum strategy ---
K_SPECTRUM_SHOTS = 5 # Total examples for each label (positive/negative)
N_SPECTRUM_CANDIDATES = 50 # Initial larger pool of candidates to search from

# Define the ranks/indices to pick from the N_SPECTRUM_CANDIDATES pool.
# This should be a list of K_SPECTRUM_SHOTS distinct indices within [0, N_SPECTRUM_CANDIDATES-1].
# Example: [0, 4, 9, 19, 29] for 5 shots from a pool of 30, picking roughly at 0%, 15%, 30%, 60%, 90%
SPECTRUM_RANKS_TO_PICK = [0, 5, 15, 25, 40] # 0-indexed ranks (1st, 6th, 16th, 26th, 41st most similar)

# --- Inference ---
# The two possible answers our LLM can give
POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"


%%writefile utils_rag.py

import pandas as pd

def build_corpus(data_path):
    """
    Builds a comprehensive corpus on-the-fly from train.csv and all examples in test.csv.
    This serves as our knowledge base for retrieval.
    """
    # This function remains unchanged.
    print("Building corpus from all available labeled data...")
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")
    corpus_data = []
    for _, row in train_df.iterrows():
        if pd.notna(row['body']):
            corpus_data.append({'text': row['body'], 'rule': row['rule'], 'label': int(row['rule_violation'])})
        for i in range(1, 3):
            if pd.notna(row[f'positive_example_{i}']):
                corpus_data.append({'text': row[f'positive_example_{i}'], 'rule': row['rule'], 'label': 1})
            if pd.notna(row[f'negative_example_{i}']):
                corpus_data.append({'text': row[f'negative_example_{i}'], 'rule': row['rule'], 'label': 0})
    for _, row in test_df.iterrows():
        for i in range(1, 3):
            if pd.notna(row[f'positive_example_{i}']):
                corpus_data.append({'text': row[f'positive_example_{i}'], 'rule': row['rule'], 'label': 1})
            if pd.notna(row[f'negative_example_{i}']):
                corpus_data.append({'text': row[f'negative_example_{i}'], 'rule': row['rule'], 'label': 0})
    corpus_df = pd.DataFrame(corpus_data)
    corpus_df = corpus_df.drop_duplicates(subset=['text', 'rule']).reset_index(drop=True)
    print(f"Corpus built successfully with {len(corpus_df)} total examples.")
    return corpus_df


# --- REVERTED TO A SIMPLER PROMPT BUILDER ---
def build_dynamic_prompt_similarity_spectrum(test_row, retrieved_pos_examples, retrieved_neg_examples):
    """
    Constructs a detailed, dynamic prompt for the LLM using a spectrum of similarity examples.
    """
    positive_examples_str = ""
    if retrieved_pos_examples:
        positive_examples_str += "Here are several positive examples (violates the rule) covering a range of similarities to the comment being evaluated:\n---\n"
        for i, example in enumerate(retrieved_pos_examples):
            positive_examples_str += f"Example {i+1}:\n'{example}'\n\n"
    else:
        positive_examples_str = "No suitable positive examples were found.\n\n"

    negative_examples_str = ""
    if retrieved_neg_examples:
        negative_examples_str += "Here are several negative examples (does NOT violate the rule) covering a range of similarities to the comment being evaluated:\n---\n"
        for i, example in enumerate(retrieved_neg_examples):
            negative_examples_str += f"Example {i+1}:\n'{example}'\n\n"
    else:
        negative_examples_str = "No suitable negative examples were found.\n\n"

    prompt = f"""
You are an expert Reddit content moderator. Your task is to determine if a comment violates a specific subreddit rule. You will be provided with examples that demonstrate the rule, chosen to cover a spectrum of relevance.

**Subreddit:** r/{test_row['subreddit']}
**Rule to Evaluate:** "{test_row['rule']}"

{positive_examples_str}
{negative_examples_str}
---
Now, carefully evaluate the following comment based on the rule and the examples provided above.

**Comment to Classify:**
"{test_row['body']}"

Does this comment violate the rule? Respond with only "Yes" or "No".
Answer:"""
    return prompt





%%writefile inference_rag.py

import os
import torch
import pandas as pd
import numpy as np
import vllm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from logits_processor_zoo.vllm import MultipleChoiceLogitsProcessor
from scipy.special import softmax
from tqdm import tqdm
import gc

from constants_rag import (
    EMBEDDING_MODEL_PATH, GENERATIVE_MODEL_PATH, DATA_PATH, 
    K_SPECTRUM_SHOTS, N_SPECTRUM_CANDIDATES, SPECTRUM_RANKS_TO_PICK, # Import new params
    POSITIVE_ANSWER, NEGATIVE_ANSWER
)
# Import the new prompt builder
from utils_rag import build_corpus, build_dynamic_prompt_similarity_spectrum

def main():
    # --- Phase 1: Dynamic Setup & Indexing (No changes here) ---
    print("--- Phase 1: Building corpus and search indexes ---")
    
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device="cuda")
    corpus_df = build_corpus(DATA_PATH)
    
    print("Creating rule-aware text for retrieval...")
    corpus_df['retrieval_text'] = "Rule: " + corpus_df['rule'] + "\nComment: " + corpus_df['text']
    
    print("Computing rule-aware embeddings for the corpus...")
    corpus_embeddings = embed_model.encode(
        corpus_df['retrieval_text'].tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda"
    )

    print("Creating rule-specific search indexes in memory...")
    rule_indexes = {}
    for rule in tqdm(corpus_df['rule'].unique(), desc="Indexing rules"):
        rule_mask = corpus_df['rule'] == rule
        rule_corpus = corpus_df[rule_mask]
        rule_embeddings_tensor = corpus_embeddings[rule_mask.to_numpy().nonzero()[0]]
        
        rule_indexes[rule] = {
            'corpus_df': rule_corpus.reset_index(drop=True),
            'embeddings': rule_embeddings_tensor
        }
    print("--- Phase 1 complete ---")

    # --- Phase 2: Inference Loop with Retrieval ---
    print("\n--- Phase 2: Generating prompts with Similarity Spectrum examples ---")
    
    test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
    prompts = []

    for _, test_row in tqdm(test_df.iterrows(), total=len(test_df), desc="Retrieving spectrum examples and building prompts"):
        query_text = test_row['body']
        rule = test_row['rule']
        
        retrieved_pos_examples, retrieved_neg_examples = [], []
        
        if rule in rule_indexes:
            index = rule_indexes[rule]
            
            query_retrieval_text = f"Rule: {rule}\nComment: {query_text}"
            query_embedding = embed_model.encode(query_retrieval_text, convert_to_tensor=True, device="cuda")
            
            # --- START OF NEW SIMILARITY SPECTRUM LOGIC ---
            
            pos_mask = index['corpus_df']['label'] == 1
            neg_mask = index['corpus_df']['label'] == 0
            
            # --- Handle Positive Examples ---
            if pos_mask.any():
                pos_corpus = index['corpus_df'][pos_mask]
                pos_embeddings = index['embeddings'][pos_mask.to_numpy()]
                
                # 1. Retrieve a large pool of N candidates
                all_pos_hits = semantic_search(query_embedding, pos_embeddings, top_k=N_SPECTRUM_CANDIDATES)[0]
                
                # 2. Select specific ranks from this pool
                selected_pos_texts = []
                for rank_idx in SPECTRUM_RANKS_TO_PICK:
                    if rank_idx < len(all_pos_hits): # Ensure the rank exists in the retrieved hits
                        hit = all_pos_hits[rank_idx]
                        selected_pos_texts.append(pos_corpus.iloc[hit['corpus_id']]['text'])
                retrieved_pos_examples = selected_pos_texts
            
            # --- Handle Negative Examples ---
            if neg_mask.any():
                neg_corpus = index['corpus_df'][neg_mask]
                neg_embeddings = index['embeddings'][neg_mask.to_numpy()]
                
                # 1. Retrieve a large pool of N candidates
                all_neg_hits = semantic_search(query_embedding, neg_embeddings, top_k=N_SPECTRUM_CANDIDATES)[0]
                
                # 2. Select specific ranks from this pool
                selected_neg_texts = []
                for rank_idx in SPECTRUM_RANKS_TO_PICK:
                    if rank_idx < len(all_neg_hits): # Ensure the rank exists in the retrieved hits
                        hit = all_neg_hits[rank_idx]
                        selected_neg_texts.append(neg_corpus.iloc[hit['corpus_id']]['text'])
                retrieved_neg_examples = selected_neg_texts

            # --- END OF NEW LOGIC ---

        # Use the new prompt builder
        prompt = build_dynamic_prompt_similarity_spectrum(test_row, retrieved_pos_examples, retrieved_neg_examples)
        prompts.append(prompt)

    print("--- Phase 2 complete ---")
    
    # --- Phase 3 & 4 (No changes here) ---
    print("\n--- Freeing VRAM by unloading the embedding model ---")
    del embed_model, corpus_embeddings, rule_indexes, corpus_df
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM freed.")

    print("\n--- Phase 3: Running LLM inference ---")

    llm = vllm.LLM(
        GENERATIVE_MODEL_PATH,
        quantization='awq',
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
        disable_log_stats=True,
        max_model_len=4096,
        disable_custom_all_reduce=True
    )
    tokenizer = llm.get_tokenizer()
    mclp = MultipleChoiceLogitsProcessor(tokenizer, choices=[POSITIVE_ANSWER, NEGATIVE_ANSWER])
    outputs = llm.generate(
        prompts,
        vllm.SamplingParams(
            skip_special_tokens=True,
            max_tokens=1,
            logits_processors=[mclp],
            logprobs=2, 
            temperature=0.0,
        ),
        use_tqdm=True,
    )
    print("--- Phase 3 complete ---")

    print("\n--- Phase 4: Processing results and creating submission file ---")
    
    logprob_yes_scores = []
    
    for out in outputs:
        logprobs_dict = {lp.decoded_token: lp.logprob for lp in out.outputs[0].logprobs[0].values()} if out.outputs[0].logprobs else {}
        logprob_yes = logprobs_dict.get(POSITIVE_ANSWER, -10.0)
        logprob_yes_scores.append(logprob_yes)

    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': logprob_yes_scores
    })
    
    submission_df['rule_violation'] = submission_df['rule_violation'].rank(pct=True)
    submission_df.to_csv("submission.csv", index=False)
    
    print("\n✅ submission.csv created successfully!")


if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "0"
    main()


!python inference_rag.py


!head submission.csv

