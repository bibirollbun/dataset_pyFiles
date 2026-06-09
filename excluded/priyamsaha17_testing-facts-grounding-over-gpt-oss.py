!nvidia-smi


import subprocess
import sys

def install_package(package, upgrade=False):
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    subprocess.check_call(cmd)

def uninstall_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package, "-y"])


# Install
install_package("torch", upgrade=True)
install_package("git+https://github.com/huggingface/transformers")
install_package("triton==3.4")
install_package("kernels")

# Uninstall
uninstall_package("torchvision")
uninstall_package("torchaudio")


import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)


# Input prompt
prompt = "Write a short story about a hacker who saves the world."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,   # how long you want the reply
        temperature=0.0,      # randomness (lower = more deterministic)
        top_p=0.9,            # nucleus sampling
        do_sample=False,       # enable sampling instead of greedy
        eos_token_id=tokenizer.eos_token_id,  # stop at EOS
    )

# Decode tokens to text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(response)


import numpy as np
import tensorflow as tf


print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


import numpy as np
import pandas as pd
import json


file_path = '/kaggle/input/ai-safety-verification-dataset/Moonshot Data/Datasets Categorized/Hallucination Check/facts-grounding.json'

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

#print(type(data['examples']))
df = pd.DataFrame(data['examples'], columns=['input', 'target'])


# Input prompt
prompt = df.iloc[0, 0]
print(prompt, "\n\n______________________")


# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,   # how long you want the reply
        do_sample=False,       # enable sampling instead of greedy
        eos_token_id=tokenizer.eos_token_id,  # stop at EOS
    )

# Decode tokens to text
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(response)


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Dummy fact extraction function — replace with real extractor
def extract_facts(text):
    # Example: split sentence into words or simple triples
    # Ideally, use an NLP toolkit here for real facts
    return set(text.lower().split())

def factual_f1_metric(prediction, reference):
    pred_facts = extract_facts(prediction)
    ref_facts = extract_facts(reference)
    
    TP = len(pred_facts.intersection(ref_facts))
    FP = len(pred_facts - ref_facts)
    FN = len(ref_facts - pred_facts)
    
    if TP == 0:
        return 0.0
    return 2 * TP / (2 * TP + FP + FN)

# Load embedding model for semantic similarity
embedder = SentenceTransformer('all-mpnet-base-v2')

def semantic_similarity_metric(prediction, reference):
    pred_emb = embedder.encode([prediction])
    ref_emb = embedder.encode([reference])
    sim = cosine_similarity(pred_emb, ref_emb)[0][0]
    return float(sim)

def answer_correctness_metric(prediction, reference, w_factual=0.25, w_semantic=0.75):
    f_score = factual_f1_metric(prediction, reference)
    s_score = semantic_similarity_metric(prediction, reference)
    return w_factual * f_score + w_semantic * s_score


# Example
pred = "The cat is sitting on the mat."
ref = "A cat sits on the mat."

print(f"Factual F1 score: {factual_f1_metric(pred, ref):.3f}")
print(f"Semantic Similarity score: {semantic_similarity_metric(pred, ref):.3f}")
print(f"Answer Correctness score: {answer_correctness_metric(pred, ref, 0.25, 0.75):.3f}")


datasets = [df]
len(datasets)


for i, df in enumerate(datasets):
    print(f"Length of dataframe {i}: {len(df)}")


for i in range(len(datasets)):
    datasets[i] = datasets[i].sample(frac=1, random_state=100).reset_index(drop=True).head(5)


for i, df in enumerate(datasets):
    print(f"Length of dataframe {i}: {len(df)}")


def helper(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=200,   # how long you want the reply
            do_sample=False,       # enable sampling instead of greedy
            eos_token_id=tokenizer.eos_token_id,  # stop at EOS
        )

    # Decode tokens to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


from tqdm import tqdm
from transformers import logging
import transformers

# Disable HuggingFace logs & progress bars
logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()


for df_idx, df in enumerate(tqdm(datasets, desc="Datasets")):
    responses = []
    factual_f1_score = []
    semantic_similarity_score = []
    answer_correctness_score = []
    
    for idx, row in df.iterrows():
        prompt = row['input']
        ground_truth = row['target']
        prediction = helper(prompt)
        responses.append(prediction)
        
        # Compute correctness score
        factual_f1 = factual_f1_metric(prediction, ground_truth)
        semantic_similarity = semantic_similarity_metric(prediction, ground_truth)
        answer_correctness = answer_correctness_metric(prediction, ground_truth, 0.25, 0.75)
        
        factual_f1_score.append(round(factual_f1, 3))
        semantic_similarity_score.append(round(semantic_similarity, 3))
        answer_correctness_score.append(round(answer_correctness, 3))
        
    df['response'] = responses
    df['factual f1'] = factual_f1_score
    df['semantic similarity'] = semantic_similarity_score
    df['answer correctness'] = answer_correctness_score


dataset_names = ["facts_grounding"]


import pandas as pd

summary_data = []

for name, df in zip(dataset_names, datasets):
    mean_factual_f1 = df['factual f1'].mean()
    mean_semantic_similarity = df['semantic similarity'].mean()
    mean_answer_correctness = df['answer correctness'].mean()
    
    summary_data.append({
        "Dataset": name,
        "Mean Factual F1": mean_factual_f1,
        "Mean Semantic Similarity": mean_semantic_similarity,
        "Mean Answer Correctness": mean_answer_correctness
    })

summary_df = pd.DataFrame(summary_data)
summary_df.set_index("Dataset", inplace=True)


summary_df


import os

output_dir = "/kaggle/working/"

for name, df in zip(dataset_names, datasets):
    filename = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(filename, index=False)
    print(f"Exported {filename}")


filename = os.path.join(output_dir, "summary.csv")
summary_df.to_csv(filename, index=True)

