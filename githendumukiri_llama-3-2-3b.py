!pip install --no-deps --no-index /kaggle/input/hf-libraries/bitsandbytes/bitsandbytes-0.45.3-py3-none-manylinux_2_24_x86_64.whl


import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, set_seed
from sentence_transformers import SentenceTransformer, util
import re
import time

set_seed(42)
start_time = time.time()

num_gpus = torch.cuda.device_count()
device_gen = "cuda:0"
device_score = "cuda:1" if num_gpus > 1 else "cuda:0"

llama_model_path = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, padding_side="left")
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_path, device_map={"": device_gen}, torch_dtype=torch.float16
)
generator = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)

quant_config = BitsAndBytesConfig(load_in_8bit=True)
llama_8b_model_path = "/kaggle/input/llama-3.1/transformers/8b-instruct/1"
llama_8b_tokenizer = AutoTokenizer.from_pretrained(llama_8b_model_path, padding_side="left")
if llama_8b_tokenizer.pad_token is None:
    llama_8b_tokenizer.pad_token = llama_8b_tokenizer.eos_token
    llama_8b_tokenizer.pad_token_id = llama_8b_tokenizer.eos_token_id
llama_8b_model = AutoModelForCausalLM.from_pretrained(
    llama_8b_model_path, quantization_config=quant_config, device_map={"": device_score}
)
scorer = pipeline("text-generation", model=llama_8b_model, tokenizer=llama_8b_tokenizer)

embed_model = SentenceTransformer('/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2', device="cpu")

df = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
topics = df["topic"].tolist()

system_prompts = [
    "You are an expert scoring essays on factual accuracy (0-9). Return only ‘[SCORE]: X’ (X = 0-9). Score 0 for nonsense, 9 for facts. Examples: 'The sky is blue due to scattering.' → [SCORE]: 9, 'Cats fly to the moon.' → [SCORE]: 0",
    "You are an expert scoring essays on style and coherence (0-9). Return only ‘[SCORE]: X’ (X = 0-9). Score 0 for disjointed text, 9 for flowing prose. Examples: 'The river flows gently.' → [SCORE]: 9, 'River. Flows. Bad.' → [SCORE]: 0",
    "You are an expert scoring essays on creativity and originality (0-9). Return only ‘[SCORE]: X’ (X = 0-9). Score 0 for dull content, 9 for unique ideas. Examples: 'Code dances like stars.' → [SCORE]: 9, 'Software works.' → [SCORE]: 0"
]

scoring_prompts = [
    "Score this essay's factual accuracy (0-9). Output only ‘[SCORE]: X’: ",
    "Score this essay's style and coherence (0-9). Output only ‘[SCORE]: X’: ",
    "Score this essay's creativity and originality (0-9). Output only ‘[SCORE]: X’: "
]

essay_prompts = [
    "Write a 100-word essay on {topic}. Ensure high accuracy and elegant prose, but no creativity.",
    "Write a 100-word essay on {topic}. Ensure high accuracy and wild creativity, but disjointed and incoherent.",
    "Write a 100-word essay on {topic}. Ensure elegant prose and wild creativity, but completely inaccurate."
]

def generate_essays(topics, num_candidates):
    all_prompts = [essay_prompts[i % 3].format(topic=topic) for i, topic in enumerate(topics)
                   for _ in range(num_candidates)]
    outputs = generator(
        all_prompts, max_new_tokens=150, do_sample=True, temperature=0.9, top_p=0.95,
        batch_size=8, truncation=True, return_full_text=False
    )
    essays = [llama_tokenizer.decode(llama_tokenizer.encode(out[0]['generated_text'].strip()), 
                                    skip_special_tokens=True).split()[:100] for out in outputs]
    return {topic: [" ".join(e) for e in essays[i*num_candidates:(i+1)*num_candidates]] 
            for i, topic in enumerate(topics)}

def generate_scores(essays, scoring_prompts, system_prompts):
    scores = []
    temperatures = [0.05, 0.1, 0.2]
    for judge_idx, (sys_prompt, prompt, temp) in enumerate(zip(system_prompts, scoring_prompts, temperatures)):
        judge_prompts = [f"{sys_prompt}\n\n{prompt}{essay}" for essay in essays]
        outputs = scorer(
            judge_prompts, max_new_tokens=100, do_sample=True, temperature=temp,
            top_k=40, top_p=0.9, batch_size=16, truncation=True, return_full_text=False
        )
        raw_outputs = [llama_8b_tokenizer.decode(llama_8b_tokenizer.encode(out[0]['generated_text'].strip()), 
                                                 skip_special_tokens=True) for out in outputs]
        judge_scores = [min(max(int(float(re.findall(r'(?i)(?:\[?\s*(?:SCORE|Score|S)\s*\]?\s*[:=]?\s*)(\d+(?:\.\d+)?)', output)[-1])), 0), 9) 
                        if re.findall(r'(?i)(?:\[?\s*(?:SCORE|Score|S)\s*\]?\s*[:=]?\s*)(\d+(?:\.\d+)?)', output) else 5 
                        for output in raw_outputs]
        scores.append(judge_scores)
    return list(zip(*scores))

num_candidates = 20
chunk_size = 200
all_essays, all_scores, all_embeddings = {}, {}, {}
for start_idx in range(0, len(topics), chunk_size):
    chunk_topics = topics[start_idx:start_idx + chunk_size]
    chunk_essays = generate_essays(chunk_topics, num_candidates)
    chunk_scores = {topic: generate_scores(essays, scoring_prompts, system_prompts) 
                    for topic, essays in chunk_essays.items()}
    chunk_embeddings = {topic: embed_model.encode(essays, convert_to_tensor=True, batch_size=8) 
                        for topic, essays in chunk_essays.items()}
    all_essays.update(chunk_essays)
    all_scores.update(chunk_scores)
    all_embeddings.update(chunk_embeddings)
    torch.cuda.empty_cache()

selected_essays, selected_scores, selected_embeddings = [], [], []
for i, topic in enumerate(topics):
    target_judge = i % 3
    candidates = [(e, s, em) for e, s, em in zip(all_essays[topic], all_scores[topic], all_embeddings[topic])
                  if not selected_embeddings or all(util.pytorch_cos_sim(em, se).item() < 0.3 for se in selected_embeddings)]
    best_score, best_candidate = -float('inf'), None
    for essay, scores, embed in candidates:
        other_scores = scores[:target_judge] + scores[target_judge+1:]
        pattern_score = scores[target_judge] - (sum(other_scores) / len(other_scores))
        if pattern_score > best_score:
            best_score, best_candidate = pattern_score, (essay, scores, embed)
    selected_essays.append(best_candidate[0])
    selected_scores.append(best_candidate[1])
    selected_embeddings.append(best_candidate[2])

avg_h = np.mean([np.var(s) for s in selected_scores])
min_v = min([np.var([s[i] for s in selected_scores]) for i in range(3)])
avg_s = max(np.mean([util.pytorch_cos_sim(e1, e2).item() 
                     for i, e1 in enumerate(selected_embeddings) 
                     for e2 in selected_embeddings[i+1:]] or [0]), 0.2)

df["essay"] = selected_essays
df[["id", "essay"]].to_csv("submission.csv", index=False)

runtime = (time.time() - start_time) / 60
print(f"Runtime: {runtime:.2f} min, avg_h: {avg_h:.2f}, min_v: {min_v:.2f}, avg_s: {avg_s:.2f}")

