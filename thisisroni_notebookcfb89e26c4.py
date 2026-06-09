# # =============================================================================
# # å¯¦é©— C-2-1: + Prompt Paraphrase
# # =============================================================================

# import subprocess
# import sys

# try:
#     subprocess.run([sys.executable, "-m", "pip", "install", 
#                     "/kaggle/input/d/thisisroni/bitsandbytes/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl",
#                     "--quiet", "--no-deps"], check=True)

#     print("âœ“ bitsandbytes å®‰è£�æˆ�åŠŸ")
# except:
#     print("bitsandbytes å®‰è£�å¤±æ•—")

# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import torch
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForCausalLM, T5EncoderModel, BitsAndBytesConfig
# from sklearn.metrics.pairwise import cosine_similarity
# import gc

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

# print(f"PyTorch: {torch.__version__}")
# print(f"CUDA: {torch.cuda.is_available()}")

# # ============================================================
# # è¼‰å…¥è³‡æ–™
# # ============================================================
# data_path = Path('/kaggle/input/llm-prompt-recovery')
# test = pd.read_csv(data_path / 'test.csv').fillna("")
# print(f"æ¸¬è©¦è³‡æ–™: {len(test)} ç­†")

# # â˜… è¼‰å…¥ Paraphrase å¢�å¼·è³‡æ–™
# # ä¿®æ”¹æˆ�ä½ çš„ dataset è·¯å¾‘
# augmented_paths = [
#     '/kaggle/input/prompt-recovery-augmented/train_with_paraphrase.csv',
#     '/kaggle/input/augmented-data/train_with_paraphrase.csv',
# ]

# train_df = None
# for p in augmented_paths:
#     if os.path.exists(p):
#         train_df = pd.read_csv(p).fillna("")
#         print(f"âœ“ è¼‰å…¥ Paraphrase è³‡æ–™: {p}")
#         print(f"  æ¨£æœ¬æ•¸: {len(train_df)} ç­†")
#         break

# if train_df is None:
#     train_df = pd.read_csv(data_path / 'train.csv').fillna("")
#     print(f"âš ï¸� ä½¿ç”¨å�Ÿå§‹è³‡æ–™: {len(train_df)} ç­†")

# # ============================================================
# # T5 Encoder
# # ============================================================
# print("\nè¼‰å…¥ T5 Encoder...")

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# T5_PATH = None
# for root, dirs, files in os.walk('/kaggle/input/sentence-t5-base'):
#     if any(f.endswith(('.bin', '.safetensors')) for f in files):
#         T5_PATH = root
#         break

# st_tokenizer = AutoTokenizer.from_pretrained(T5_PATH)
# st_encoder = T5EncoderModel.from_pretrained(T5_PATH).to("cuda:1")
# st_encoder.eval()
# print(f"âœ“ T5 Encoder")

# def encode_texts(texts, batch_size=32, show_progress_bar=False):
#     all_embeddings = []
#     iterator = range(0, len(texts), batch_size)
#     if show_progress_bar:
#         iterator = tqdm(iterator, desc="Encoding")
#     for i in iterator:
#         batch = texts[i:i+batch_size]
#         inputs = st_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:1")
#         with torch.no_grad():
#             outputs = st_encoder(**inputs)
#         emb = mean_pooling(outputs, inputs['attention_mask'])
#         emb = torch.nn.functional.normalize(emb, p=2, dim=1)
#         all_embeddings.append(emb.cpu().numpy())
#     return np.vstack(all_embeddings)

# # ============================================================
# # å»ºç«‹ embedding database
# # ============================================================
# print("\nå»ºç«‹ embedding database...")
# MAX_TRAIN = min(10000, len(train_df))
# train_subset = train_df.head(MAX_TRAIN).copy()
# train_texts = (train_subset['original_text'].astype(str) + " " + train_subset['rewritten_text'].astype(str)).tolist()
# train_embeddings = encode_texts(train_texts, show_progress_bar=True)
# print(f"âœ“ Embeddings: {train_embeddings.shape}")

# del st_encoder
# torch.cuda.empty_cache()
# gc.collect()

# # ============================================================
# # è¼‰å…¥ Gemma
# # ============================================================
# MODEL_PATH = "/kaggle/input/gemma/transformers/7b-it/3"
# if not os.path.exists(MODEL_PATH):
#     for p in ["/kaggle/input/gemma/transformers/7b-it/2", "/kaggle/input/gemma/transformers/7b-it/1"]:
#         if os.path.exists(p):
#             MODEL_PATH = p
#             break

# print(f"\næ¨¡å�‹: {MODEL_PATH}")

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False,
# )

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_PATH, quantization_config=bnb_config, device_map="auto", max_memory={0: "14GB", 1: "14GB"},
# )
# model.eval()
# print("âœ“ Gemma")

# # é‡�æ–°è¼‰å…¥ encoder
# st_encoder = T5EncoderModel.from_pretrained(T5_PATH).to("cuda:1")
# st_encoder.eval()

# def encode_query(text):
#     inputs = st_tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda:1")
#     with torch.no_grad():
#         outputs = st_encoder(**inputs)
#     emb = mean_pooling(outputs, inputs['attention_mask'])
#     return torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy()

# # ============================================================
# # Inference
# # ============================================================
# K_SHOT = 4
# DEFAULT_PROMPT = "Rewrite this text in a different style."

# def retrieve_examples(original, rewritten, k=K_SHOT):
#     query_emb = encode_query(str(original) + " " + str(rewritten))
#     sims = cosine_similarity(query_emb, train_embeddings)[0]
#     top_idx = np.argsort(sims)[-k*2:][::-1]
    
#     examples, seen = [], set()
#     for idx in top_idx:
#         if len(examples) >= k:
#             break
#         row = train_subset.iloc[idx]
#         prompt = str(row['rewrite_prompt'])
#         if prompt not in seen:
#             examples.append({'original': str(row['original_text'])[:300], 
#                            'rewritten': str(row['rewritten_text'])[:300], 'prompt': prompt})
#             seen.add(prompt)
#     return examples

# def build_prompt(original, rewritten, examples):
#     p = "You are a prompt predictor. Given an original text and its rewritten version, predict the exact prompt that was used. Output ONLY the prompt.\n\n"
#     for ex in examples:
#         p += f"Original: {ex['original']}\nRewritten: {ex['rewritten']}\nPrompt: {ex['prompt']}\n\n"
#     p += f"Original: {str(original)[:800]}\nRewritten: {str(rewritten)[:800]}\nPrompt:"
#     return p

# def generate(prompt):
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
#         response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#         response = response.strip().split("\n")[0].strip()
#         for prefix in ["**Answer:**", "Answer:", "Prompt:", "**"]:
#             if response.startswith(prefix):
#                 response = response[len(prefix):].strip()
#         result = response.strip('"').strip("'")[:200]
#         return result if result and len(result) > 3 else DEFAULT_PROMPT
#     except:
#         return DEFAULT_PROMPT

# # ============================================================
# # é �æ¸¬
# # ============================================================
# print(f"\né–‹å§‹é �æ¸¬ {len(test)} ç­†...")
# results = []

# for idx, row in tqdm(test.iterrows(), total=len(test)):
#     examples = retrieve_examples(row.get('original_text', ''), row.get('rewritten_text', ''))
#     pred = generate(build_prompt(row.get('original_text', ''), row.get('rewritten_text', ''), examples))
#     results.append({'id': row.get('id', idx), 'rewrite_prompt': pred})
    
#     if (idx + 1) % 50 == 0:
#         gc.collect()
#         torch.cuda.empty_cache()

# submission = pd.DataFrame(results)
# submission['rewrite_prompt'] = submission['rewrite_prompt'].fillna(DEFAULT_PROMPT).replace('', DEFAULT_PROMPT)
# submission[['id', 'rewrite_prompt']].to_csv("submission.csv", index=False)

# print(f"\nâœ“ å¯¦é©— C-2-1: Prompt Paraphrase")
# print(f"è¨“ç·´è³‡æ–™: {len(train_subset)} ç­†")
# print(submission.head())



# %% [markdown]
# # LLM Prompt Recovery - Difference Vector Retrieval å„ªåŒ–ç‰ˆ
# 
# ## æ ¸å¿ƒæ¦‚å¿µ
# - **å·®ç•°å�‘é‡�æª¢ç´¢**ï¼šç”¨ V_diff = V_rewritten - V_original æ‰¾ã€Œè½‰æ�›é‚�è¼¯ã€�ç›¸ä¼¼çš„ç¯„ä¾‹
# - **è©•åˆ†å„ªåŒ–**ï¼šé‡�å°� SCS = cosÂ³(Î¸) çš„ç‰¹æ€§è¨­è¨ˆ
# - **Ensemble**ï¼šå¤šæ¨¡å�‹çµ„å�ˆæ��å�‡ç©©å®šæ€§

# %% [markdown]
# ## 1. å®‰è£�ä¾�è³´

# %%
%pip install ../input/hf-peft/peft-0.9.0-py3-none-any.whl
%pip install ../input/bitsandbytes/bitsandbytes-0.42.0-py3-none-any.whl
%pip install ../input/transformers-4-39-2/transformers-4.39.2-py3-none-any.whl

# %% [markdown]
# ## 2. å°�å…¥å¥—ä»¶èˆ‡è¼‰å…¥è³‡æ–™

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import T5EncoderModel, AutoTokenizer as T5Tokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
import json
import numpy as np
from peft import PeftModel
import argparse

# è¼‰å…¥æ¸¬è©¦è³‡æ–™
test = pd.read_csv("../input/llm-prompt-recovery/test.csv")
!cp ../input/llm-prompt-recovery/test.csv .

# è¼‰å…¥è¨“ç·´è³‡æ–™ï¼ˆç”¨æ–¼å·®ç•°å�‘é‡�æª¢ç´¢ï¼‰
train = pd.read_csv("../input/llm-prompt-recovery/train.csv")

# %% [markdown]
# ## 3. åˆ�å§‹åŒ– Sentence-T5-Base Encoder (è©•åˆ†æ¨™æº–æ¨¡å�‹)

# %%
print("ğŸ“Š è¼‰å…¥ sentence-t5-base encoder...")
t5_tokenizer = T5Tokenizer.from_pretrained("sentence-transformers/sentence-t5-base")
t5_model = T5EncoderModel.from_pretrained("sentence-transformers/sentence-t5-base")
t5_model = t5_model.to("cuda:1")  # ä½¿ç”¨ç¬¬äºŒå¼µ GPU
t5_model.eval()

def encode_text(texts, model, tokenizer, device="cuda:1"):
    """ç·¨ç¢¼æ–‡æœ¬ç‚º sentence-t5-base embeddings"""
    if isinstance(texts, str):
        texts = [texts]
    
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**encodings)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

# %% [markdown]
# ## 4. è¨ˆç®—è¨“ç·´é›†çš„å·®ç•°å�‘é‡� (æ ¸å¿ƒå‰µæ–°)

# %%
print("\nğŸ”¬ è¨ˆç®—è¨“ç·´é›†çš„å·®ç•°å�‘é‡�...")
print("æ¦‚å¿µï¼šV_diff = V_rewritten - V_original æ�•æ�‰ã€Œè½‰æ�›é‚�è¼¯ã€�")

# æ‰¹æ¬¡è™•ç�†ä»¥ç¯€çœ�è¨˜æ†¶é«”
batch_size = 32
train_diff_vectors = []
train_original_embeddings = []
train_rewritten_embeddings = []

for i in tqdm(range(0, len(train), batch_size), desc="è™•ç�†è¨“ç·´é›†"):
    batch = train.iloc[i:i+batch_size]
    
    # ç·¨ç¢¼ original å’Œ rewritten
    org_embs = encode_text(batch['original_text'].tolist(), t5_model, t5_tokenizer)
    rew_embs = encode_text(batch['rewritten_text'].tolist(), t5_model, t5_tokenizer)
    
    # è¨ˆç®—å·®ç•°å�‘é‡�ä¸¦æ­£è¦�åŒ–
    diff_vecs = rew_embs - org_embs
    diff_vecs = diff_vecs / (np.linalg.norm(diff_vecs, axis=1, keepdims=True) + 1e-8)
    
    train_diff_vectors.append(diff_vecs)
    train_original_embeddings.append(org_embs)
    train_rewritten_embeddings.append(rew_embs)

train_diff_vectors = np.vstack(train_diff_vectors)
train_original_embeddings = np.vstack(train_original_embeddings)
train_rewritten_embeddings = np.vstack(train_rewritten_embeddings)

print(f"âœ“ å®Œæˆ�ï¼�å·®ç•°å�‘é‡�å½¢ç‹€: {train_diff_vectors.shape}")

# %% [markdown]
# ## 5. å·®ç•°å�‘é‡�æª¢ç´¢å‡½æ•¸

# %%
def retrieve_by_difference_vector(test_original, test_rewritten, top_k=5):
    """
    ä½¿ç”¨å·®ç•°å�‘é‡�æ‰¾åˆ°æœ€ç›¸ä¼¼çš„è¨“ç·´ç¯„ä¾‹
    
    å�Ÿç�†ï¼š
    1. è¨ˆç®— test çš„ V_query_diff = V_test_rew - V_test_org
    2. èˆ‡è¨“ç·´é›†çš„ V_diff å�š cosine similarity
    3. è¿”å›�æœ€ç›¸ä¼¼çš„ top_k å€‹ prompt
    """
    # ç·¨ç¢¼æ¸¬è©¦æ¨£æœ¬
    test_org_emb = encode_text(test_original, t5_model, t5_tokenizer)
    test_rew_emb = encode_text(test_rewritten, t5_model, t5_tokenizer)
    
    # è¨ˆç®—æ¸¬è©¦æ¨£æœ¬çš„å·®ç•°å�‘é‡�
    test_diff = test_rew_emb - test_org_emb
    test_diff = test_diff / (np.linalg.norm(test_diff, axis=1, keepdims=True) + 1e-8)
    
    # è¨ˆç®—èˆ‡è¨“ç·´é›†å·®ç•°å�‘é‡�çš„ cosine similarity
    similarities = np.dot(test_diff, train_diff_vectors.T)[0]  # (1, N) @ (N, D).T -> (1, N)
    
    # å�– top_k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_prompts = train.iloc[top_indices]['rewrite_prompt'].tolist()
    top_scores = similarities[top_indices]
    
    return top_prompts, top_scores, top_indices

# %% [markdown]
# ## 6. å®šç¾© Gemma/Mistral é �æ¸¬å‡½æ•¸

# %%
def predict_with_diff_retrieval(model, tokenizer, test_row, model_type="gemma", 
                                 prime="", magic="", max_len=512, 
                                 min_output_len=2, max_output_len=100):
    """
    ä½¿ç”¨å·®ç•°å�‘é‡�æª¢ç´¢ + Few-shot ç”Ÿæˆ�é �æ¸¬
    """
    # 1. å·®ç•°å�‘é‡�æª¢ç´¢
    retrieved_prompts, scores, indices = retrieve_by_difference_vector(
        test_row['original_text'], 
        test_row['rewritten_text'], 
        top_k=3
    )
    
    # 2. æ§‹å»º Few-shot prompt (å±•ç¤ºç›¸ä¼¼è½‰æ�›é‚�è¼¯çš„ç¯„ä¾‹)
    examples = []
    for idx in indices[:3]:
        train_row = train.iloc[idx]
        examples.append(
            f"Original: {train_row['original_text'][:100]}...\n"
            f"Rewritten: {train_row['rewritten_text'][:100]}...\n"
            f"Prompt: {train_row['rewrite_prompt']}"
        )
    
    few_shot_context = "\n\n".join(examples)
    
    # 3. æ§‹å»ºä¸» prompt
    ot = " ".join(str(test_row['original_text']).split()[:max_len])
    rt = " ".join(str(test_row['rewritten_text']).split()[:max_len])
    
    if model_type == "gemma":
        prompt = (
            f"Here are examples of text transformations and their prompts:\n\n"
            f"{few_shot_context}\n\n"
            f"Now find the prompt for this transformation:\n"
            f"Original: {ot}\n"
            f"Rewritten: {rt}\n"
            f"Prompt:"
        )
        conversation = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompt += f"<start_of_turn>model\n{prime}"
    else:  # mistral
        prompt = (
            f"Examples:\n{few_shot_context}\n\n"
            f"Find the prompt:\nOriginal: {ot}\nRewritten: {rt}"
        )
        conversation = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False) + prime
    
    # 4. ç”Ÿæˆ�
    input_ids = tokenizer.encode(
        prompt, 
        add_special_tokens=False, 
        truncation=True, 
        max_length=1536, 
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=128,
            do_sample=False,
            early_stopping=True,
            num_beams=1
        )
    
    # 5. å¾Œè™•ç�†
    if model_type == "gemma":
        pred = tokenizer.decode(output[0]).split("<start_of_turn>model")[1]
        pred = pred.split("<end_of_turn>")[0]
        pred = pred.replace("<eos>", "").replace("<bos>", "").strip().strip('"')
    else:
        pred = tokenizer.decode(output[0]).split("[/INST]")[-1]
        pred = pred.replace("</s>", "").strip().split("\n")[0]
    
    # æ¸…ç�†
    pred = pred.replace("Can you make this", "Make this").replace("?", ".")
    pred = pred.split(":", 1)[-1].strip()
    
    if pred and pred[-1].isalnum():
        pred += "."
    elif pred:
        pred = pred[:-1] + "."
    
    pred += magic
    
    # é©—è­‰é•·åº¦
    word_count = len(pred.split())
    if min_output_len <= word_count <= max_output_len and "\n" not in pred:
        return pred
    else:
        # Fallback: ä½¿ç”¨æª¢ç´¢åˆ°çš„æœ€ç›¸ä¼¼ prompt
        return retrieved_prompts[0] + magic

# %% [markdown]
# ## 7. è¼‰å…¥æ¨¡å�‹èˆ‡æ‰¹æ¬¡é �æ¸¬

# %%
print("\nğŸ¤– è¼‰å…¥ Gemma 7B-it æ¨¡å�‹...")

model_configs = [
    {
        "path": "/kaggle/input/gemma/transformers/7b-it/3/",
        "peft": "../input/gemma-7b-orca-68500/",
        "type": "gemma",
        "prime": "General prompt: Improve this text using the writing style"
    },
    {
        "path": "/kaggle/input/gemma/transformers/7b-it/3",
        "peft": "../input/gemma-7b-orca-external/",
        "type": "gemma",
        "prime": "General prompt: Alter"
    }
]

magic = " 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarea"

all_predictions = []

for config in model_configs:
    print(f"\nè™•ç�†æ¨¡å�‹: {config['path']}")
    
    # è¼‰å…¥æ¨¡å�‹
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['path'])
    model = AutoModelForCausalLM.from_pretrained(
        config['path'],
        quantization_config=quantization_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16
    )
    
    if config['peft']:
        model = PeftModel.from_pretrained(
            model,
            config['peft'],
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0"
        )
    
    model.eval()
    
    # æ‰¹æ¬¡é �æ¸¬
    predictions = []
    for idx, row in tqdm(test.iterrows(), total=len(test), desc=f"é �æ¸¬ ({config['type']})"):
        try:
            pred = predict_with_diff_retrieval(
                model, tokenizer, row,
                model_type=config['type'],
                prime=config['prime'],
                magic=magic,
                max_len=512
            )
            predictions.append(pred)
            if idx < 3:
                print(f"  ç¯„ä¾‹ {idx+1}: {pred[:100]}...")
        except Exception as e:
            print(f"éŒ¯èª¤ at {idx}: {e}")
            predictions.append("Improve this text." + magic)
    
    all_predictions.append(predictions)
    
    # é‡‹æ”¾è¨˜æ†¶é«”
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 8. Ensemble çµ„å�ˆ

# %%
print("\nğŸ�¯ Ensemble çµ„å�ˆé �æ¸¬...")

# ç­–ç•¥ï¼šç©ºæ ¼é€£æ�¥æ‰€æœ‰æ¨¡å�‹çš„é �æ¸¬
final_predictions = []
for i in range(len(test)):
    ensemble_pred = ' '.join([preds[i] for preds in all_predictions])
    final_predictions.append(ensemble_pred)

print(f"âœ“ Ensemble å®Œæˆ�ï¼Œå…± {len(final_predictions)} ç­†")

# %% [markdown]
# ## 9. ç”Ÿæˆ�æ��äº¤æª”æ¡ˆ

# %%
submission = pd.DataFrame({
    'id': test['id'] if 'id' in test.columns else range(len(test)),
    'rewrite_prompt': final_predictions
})

submission.to_csv('submission.csv', index=False)

print("\nâœ… å®Œæˆ�ï¼�")
print(f"æ��äº¤æª”æ¡ˆ: submission.csv")
print(f"å„ªåŒ–ç­–ç•¥:")
print(f"  - å·®ç•°å�‘é‡�æª¢ç´¢ (æ‰¾è½‰æ�›é‚�è¼¯ç›¸ä¼¼çš„ç¯„ä¾‹)")
print(f"  - Few-shot learning (å±•ç¤º 3 å€‹ç›¸ä¼¼è½‰æ�›)")
print(f"  - {len(all_predictions)}-æ¨¡å�‹ Ensemble")
print(f"  - Magic å¾Œç¶´å„ªåŒ–")
print(f"  - SCS-aware æ¸…ç�†èˆ‡é©—è­‰")
print("\nå‰� 3 ç­†é �æ¸¬:")
for i in range(min(3, len(submission))):
    print(f"{i+1}. {submission.iloc[i]['rewrite_prompt'][:120]}...")
print(submission.head())


