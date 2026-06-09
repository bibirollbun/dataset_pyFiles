import kagglehub

# Download latest version
path = kagglehub.model_download("keras/gemma/keras/gemma_1.1_instruct_2b_en")

print("Path to model files:", path)


import kagglehub

# Download latest version
path = kagglehub.model_download("google/gemma/transformers/7b-it")

print("Path to model files:", path)


import kagglehub

# Download latest version
path = kagglehub.model_download("google/gemma/transformers/2b-it")

print("Path to model files:", path)


import kagglehub

# Download latest version
path = kagglehub.model_download("google/sentence-t5/tensorFlow2/st5-base")

print("Path to model files:", path)


import kagglehub

# Download latest version
path = kagglehub.model_download("akinduhiman/all-minilm-l6-v2/transformers/default")

print("Path to model files:", path)


# ===== ç¬¬ä¸€æ­¥ï¼šè¨­ç½®ç’°å¢ƒè®Šæ•¸ï¼ˆå¿…é ˆåœ¨æœ€é–‹é ­ï¼‰=====
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pandas as pd
import numpy as np
from collections import Counter
import torch
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ğŸš€ LLM Prompt Recovery - è‡ªè¨‚ç‰ˆæœ¬")
print("="*60)
print("âœ“ ä½¿ç”¨ sentence-t5-base ç›¸ä¼¼åº¦æ¨¡å�‹")
print("âœ“ ä½¿ç”¨è‡ªè¨‚è³‡æ–™é›†")



# ===== é…�ç½®ï¼ˆä½ çš„è·¯å¾‘ï¼‰=====
DATA_FILE = "../input/new-data/prompts_0_500_wiki_first_para_3000.csv"
MODEL_PATH = "/kaggle/input/m/google/gemma/transformers/2b-it/3"
SENTENCE_T5_PATH = "/kaggle/input/sentence-t5/tensorflow2/st5-base/1"

N_CANDIDATES = 300  # å€™é�¸æ•¸ï¼ˆå�¯èª¿æ•´ 200-400ï¼‰
BATCH_SIZE = 8      # æ‰¹æ¬¡å¤§å°�
MAX_TOKENS = 100    # ç”Ÿæˆ�é•·åº¦
TEST_SIZE = 0.2     # æ¸¬è©¦é›†æ¯”ä¾‹ï¼ˆ20%ï¼‰
TEST_SAMPLES = 2  # â†� åŠ é€™è¡Œï¼šå�ªæ¸¬è©¦ 10 å€‹æ¨£æœ¬
print(f"\né…�ç½®:")
print(f"  è³‡æ–™æª”æ¡ˆ: {DATA_FILE}")
print(f"  æ¨¡å�‹è·¯å¾‘: {MODEL_PATH}")
print(f"  å€™é�¸æ•¸: {N_CANDIDATES}")
print(f"  æ‰¹æ¬¡å¤§å°�: {BATCH_SIZE}")
print(f"  æ¸¬è©¦é›†æ¯”ä¾‹: {TEST_SIZE}")

print(f"  æ¸¬è©¦æ¨£æœ¬æ•¸: {TEST_SAMPLES}")  # â†� åŠ é€™è¡Œ


# ===== GPU æª¢æŸ¥ =====
print(f"\n[GPU æª¢æŸ¥]")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print(f"âœ“ GPU: {torch.cuda.get_device_name(0)}")
    print(f"âœ“ è¨˜æ†¶é«”: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    device = torch.device("cpu")
    print("âš ï¸�  ä½¿ç”¨ CPU")





# ===== 1. è¼‰å…¥è³‡æ–™ =====
print(f"\n{'='*60}")
print("[1/4] è¼‰å…¥è³‡æ–™")
print("="*60)

data = pd.read_csv(DATA_FILE)
train, test = train_test_split(data, test_size=TEST_SIZE, random_state=42)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# â­� ç¢ºä¿�æ¸¬è©¦é›†æœ‰ idï¼ˆå¦‚æ�œæ²’æœ‰å°±å‰µå»ºï¼‰
if 'id' not in test.columns:
    test['id'] = test.index
    print("  âš ï¸�  æ¸¬è©¦é›†æ²’æœ‰ idï¼Œå·²è‡ªå‹•å‰µå»º")

# â­� ç¢ºèª� id å·²ç¶“å­˜åœ¨
print(f"âœ“ æœ¬åœ°æ¸¬è©¦")
print(f"  è¨“ç·´: {len(train)}, æ¸¬è©¦: {len(test)}")
print(f"  æ¸¬è©¦é›†æ¬„ä½�: {list(test.columns)}")
print(f"  æ¸¬è©¦é›† id ç¯„åœ�: {test['id'].min()} - {test['id'].max()}")
# é™�åˆ¶æ¸¬è©¦æ¨£æœ¬æ•¸
test = test.head(TEST_SAMPLES).reset_index(drop=True)  # â†� åŠ é€™è¡Œ
print(f"âœ“ è¨“ç·´é›†: {len(train)} ç­†")
print(f"âœ“ æ¸¬è©¦é›†: {len(test)} ç­†")

# ===== 2. å‰µå»ºå€™é�¸ =====
print(f"\n{'='*60}")
print("[2/4] å‰µå»ºå€™é�¸ Prompts")
print("="*60)

prompt_counts = Counter(train['rewrite_prompt'])
candidates = [p for p, _ in prompt_counts.most_common(N_CANDIDATES)]
coverage = sum(c for _, c in prompt_counts.most_common(N_CANDIDATES)) / len(train) * 100

print(f"âœ“ è¨“ç·´é›† prompts: {len(prompt_counts)} ç¨®")
print(f"âœ“ é�¸æ“‡å€™é�¸: {len(candidates)} å€‹")
print(f"âœ“ è¦†è“‹ç�‡: {coverage:.1f}%")

if coverage < 50:
    print(f"âš ï¸�  è¦†è“‹ç�‡è¼ƒä½�ï¼Œå»ºè­°å¢�åŠ åˆ° {min(len(prompt_counts), 400)}")

# é¡¯ç¤ºå‰�å¹¾å€‹
print(f"\nå‰� 5 å€‹æœ€å¸¸è¦‹ prompts:")
for i, (prompt, count) in enumerate(prompt_counts.most_common(5), 1):
    pct = (count / len(train)) * 100
    print(f"  {i}. ({pct:.1f}%) {prompt[:60]}...")


print(f"\n{'='*60}")
print("[3/4] è¼‰å…¥æ¨¡å�‹")
print("="*60)

# Tokenizer
print("è¼‰å…¥ tokenizer...")
try:
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=False,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    print("  âœ“ AutoTokenizer æˆ�åŠŸ")
    
except Exception as e:
    print(f"  âœ— Tokenizer è¼‰å…¥å¤±æ•—: {str(e)[:100]}")
    raise


print("\nè¼‰å…¥ç”Ÿæˆ�æ¨¡å�‹ (Gemma)...")
from transformers import AutoModelForCausalLM

generator = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
    trust_remote_code=True
)
generator.eval()
print("  âœ“ Gemma è¼‰å…¥æˆ�åŠŸ")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    print(f"  âœ“ GPU è¨˜æ†¶é«”: {allocated:.2f} GB")

# ç›¸ä¼¼åº¦æ¨¡å�‹ - å¾�æœ¬åœ°è¼‰å…¥ MiniLM
print("\nè¼‰å…¥ç›¸ä¼¼åº¦æ¨¡å�‹ (all-MiniLM-L6-v2 æœ¬åœ°)...")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ä½ çš„æœ¬åœ°æ¨¡å�‹è·¯å¾‘ï¼ˆæ ¹æ“šæˆªåœ–çµ�æ§‹ï¼‰
MINILM_PATH = "/kaggle/input/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"  # â†� æ³¨æ„�é€™è£¡ï¼�

try:
    print(f"  å¾� {MINILM_PATH} è¼‰å…¥...")
    
    # å¾�æœ¬åœ°è¼‰å…¥
    sim_model = SentenceTransformer(MINILM_PATH)
    
    # æ¸¬è©¦
    test_emb = sim_model.encode(["test"], show_progress_bar=False)
    
    print(f"  âœ“ all-MiniLM-L6-v2 è¼‰å…¥æˆ�åŠŸ (æœ¬åœ°)")
    print(f"  âœ“ ç·¨ç¢¼ç¶­åº¦: {test_emb.shape[1]}")
    print(f"  âœ“ æ ¼å¼�: PyTorch (model.safetensors)")
    print(f"  âœ“ ä¾†æº�: Kaggle Dataset (å®Œå…¨é›¢ç·š)")
    
except Exception as e:
    print(f"  âœ— è¼‰å…¥å¤±æ•—: {str(e)[:200]}")
    
    # å¦‚æ�œé‚„æ˜¯å¤±æ•—ï¼Œå�¯èƒ½è·¯å¾‘ä¸�å°�ï¼Œè©¦è©¦å…¶ä»–å�¯èƒ½
    print("\n  å˜—è©¦å…¶ä»–è·¯å¾‘...")
    
    import os
    possible_paths = [
        "/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2",
        "/kaggle/input/all-minilm-l6-v2",
        "/kaggle/input/all-minilm-l6-v2/default/1/all-MiniLM-L6-v2",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"  å˜—è©¦: {path}")
                sim_model = SentenceTransformer(path)
                test_emb = sim_model.encode(["test"], show_progress_bar=False)
                print(f"  âœ“ æˆ�åŠŸï¼�ä½¿ç”¨è·¯å¾‘: {path}")
                print(f"  âœ“ ç·¨ç¢¼ç¶­åº¦: {test_emb.shape[1]}")
                break
            except Exception as e2:
                print(f"  âœ— å¤±æ•—: {str(e2)[:60]}")
                continue



# ===== 4. é �æ¸¬ =====
print(f"\n{'='*60}")
print(f"[4/4] é �æ¸¬ {len(test)} å€‹æ¨£æœ¬")
print(f"\n{'='*60}")

def predict_sample(sample):
    """é �æ¸¬å–®å€‹æ¨£æœ¬"""
    all_inputs = [
        f"<start_of_turn>user\n{p}: {sample['original_text']}\n<end_of_turn>\n<start_of_turn>model\n"
        for p in candidates
    ]
    
    generated = []
    num_batches = (len(all_inputs) - 1) // BATCH_SIZE + 1
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_inputs))
        batch = all_inputs[start_idx:end_idx]
        
        inputs = tokenizer(batch, return_tensors="pt", 
                          truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(generator.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = generator.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        for j, output in enumerate(outputs):
            gen = tokenizer.decode(output, skip_special_tokens=True)
            if len(batch[j]) < len(gen):
                gen = gen[len(batch[j]):].strip()
            generated.append(gen)
    
    t_emb = sim_model.encode([sample['rewritten_text']], show_progress_bar=False)
    g_embs = sim_model.encode(generated, show_progress_bar=False)
    sims = cosine_similarity(t_emb, g_embs)[0]
    
    best_idx = np.argmax(sims)
    return candidates[best_idx], sims[best_idx]

# é–‹å§‹é �æ¸¬
predictions = []
test_results = []
start_time = time.time()

print(f"\né–‹å§‹é �æ¸¬...")

for idx, sample in test.iterrows():
    # é€²åº¦é¡¯ç¤º
    current = idx + 1
    elapsed = time.time() - start_time
    
    if current > 1:
        avg_time = elapsed / (current - 1)
        eta = (len(test) - current + 1) * avg_time
        print(f"  [{current}/{len(test)}] - "
              f"å·²ç”¨æ™‚: {elapsed:.1f}ç§’ - "
              f"é �ä¼°å‰©é¤˜: {eta:.1f}ç§’")
    else:
        print(f"  [{current}/{len(test)}] - é–‹å§‹...")
    
    # é �æ¸¬
    pred, score = predict_sample(sample)
    
    # â­� å®‰å…¨ç�²å�– id
    if 'id' in sample:
        sample_id = sample['id']
    else:
        sample_id = idx
        print(f"    âš ï¸�  æ¨£æœ¬ {idx} æ²’æœ‰ idï¼Œä½¿ç”¨ index")
    
    predictions.append({
        'id': sample_id,
        'rewrite_prompt': pred
    })
    
    # å¦‚æ�œæœ‰çœŸå¯¦æ¨™ç±¤
    if 'rewrite_prompt' in sample:
        true = sample['rewrite_prompt']
        correct = (pred == true)
        test_results.append({
            'id': sample_id,
            'pred': pred,
            'true': true,
            'correct': correct,
            'score': score
        })
        
        status = "âœ“" if correct else "âœ—"
        print(f"    {status} ç›¸ä¼¼åº¦: {score:.3f}")
        if not correct:
            print(f"      é �æ¸¬: {pred[:50]}...")
            print(f"      çœŸå¯¦: {true[:50]}...")

total_time = time.time() - start_time

print(f"\nâœ“ é �æ¸¬å®Œæˆ�ï¼�")
print(f"  ç¸½æ™‚é–“: {total_time:.1f} ç§’ ({total_time/60:.2f} åˆ†é�˜)")
print(f"  å¹³å�‡æ™‚é–“: {total_time/len(test):.1f} ç§’/æ¨£æœ¬")

# ===== ç”Ÿæˆ�æ��äº¤æª”æ¡ˆ =====
print(f"\n{'='*60}")
print("ç”Ÿæˆ� submission.csv")
print("="*60)

# æª¢æŸ¥ predictions
print(f"æª¢æŸ¥ predictions:")
print(f"  æ•¸é‡�: {len(predictions)}")

if len(predictions) > 0:
    print(f"  ç¬¬ä¸€ç­†: {predictions[0]}")
    
    # å‰µå»º DataFrame
    submission = pd.DataFrame(predictions)
    
    # ç¢ºèª�æ¬„ä½�
    print(f"  DataFrame æ¬„ä½�: {list(submission.columns)}")
    
    # é�¸æ“‡éœ€è¦�çš„æ¬„ä½�
    submission = submission[['id', 'rewrite_prompt']]
    
    # â­� ä¿®æ­£ï¼šå°‡ id è½‰æ�›æˆ�æ­£ç¢ºæ ¼å¼�
    def format_id(id_num):
        """
        å°‡æ•¸å­— id è½‰æ�›æˆ�å­—ä¸²æ ¼å¼�
        ä¾‹å¦‚: 0 -> "000aaa", 1 -> "001aaa", 999 -> "999aaa"
        """
        # ç”Ÿæˆ� 3 ä½�æ•¸å­— + 3 å€‹å­—æ¯�
        num_part = f"{int(id_num):03d}"
        
        # æ ¹æ“š id ç”Ÿæˆ�ä¸�å�Œçš„å­—æ¯�çµ„å�ˆ
        # ç°¡å–®æ–¹å¼�ï¼šå¾ªç’°ä½¿ç”¨ aaa, bbb, ccc...
        letter_index = (int(id_num) // 100) % 26  # æ¯� 100 å€‹æ�›ä¸€å€‹å­—æ¯�
        letter = chr(97 + letter_index)  # a=97, b=98, c=99...
        letter_part = letter * 3
        
        return f"{num_part}{letter_part}"
    
    submission['id'] = submission['id'].apply(format_id)
    
    # â­� ä¿®æ­£ï¼šå„²å­˜æ™‚å�ªå°�åŒ…å�«é€—è™Ÿçš„å…§å®¹åŠ å¼•è™Ÿ
    submission.to_csv('submission.csv', index=False, quoting=2)  # quoting=2 = QUOTE_NONNUMERIC æ”¹æˆ� QUOTE_MINIMAL
    
    print(f"\nâœ“ submission.csv å·²ç”Ÿæˆ�")
    print(f"  æ¨£æœ¬æ•¸: {len(submission)}")
    print(f"  æ¬„ä½�: {list(submission.columns)}")
    
    # æ ¼å¼�é©—è­‰
    print(f"\næ ¼å¼�é©—è­‰:")
    print(f"  âœ“ æ¬„ä½�: {list(submission.columns)}")
    print(f"  âœ“ id é¡�å�‹: {submission['id'].dtype}")
    print(f"  âœ“ id ç¯„ä¾‹: {submission['id'].iloc[:5].tolist()}")
    
    if submission.isnull().any().any():
        print(f"  âœ— æœ‰ç©ºå€¼")
    else:
        print(f"  âœ“ ç„¡ç©ºå€¼")
    
    # é �è¦½
    print(f"\né �è¦½:")
    print(submission)
    
    # çµ±è¨ˆ
    from collections import Counter
    pred_counts = Counter(submission['rewrite_prompt'])
    print(f"\né �æ¸¬çµ±è¨ˆ:")
    print(f"  ä¸�å�Œ prompts: {len(pred_counts)}")
    
    print(f"\n{'='*60}")
    print("âœ… å®Œæˆ�ï¼�")
    print("="*60)
    
    # â­� æª¢æŸ¥å¯¦éš› CSV æ ¼å¼�
    print(f"\nCSV æª”æ¡ˆå…§å®¹:")
    with open('submission.csv', 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # é¡¯ç¤ºå‰� 5 è¡Œ
                print(f"  {line.rstrip()}")
    
    print(f"\nğŸ“‚ å�¯ä»¥ä¸‹è¼‰ submission.csv äº†ï¼�")
    
else:
    print("  âœ— predictions æ˜¯ç©ºçš„ï¼�")

