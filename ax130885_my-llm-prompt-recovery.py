# 1. ç§»é™¤ TensorFlow è¡�çª�
!pip uninstall -y tensorflow tensorflow-io tensorflow-estimator tensorboard

# 2. è¨­å®šç’°å¢ƒè®Šæ•¸
import os
os.environ["USE_TORCH"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 3. å®‰è£� bitsandbytes
import sys
import subprocess

BNB_WHL_PATH = '/kaggle/input/llm-prompt-recovery-dependency/bitsandbytes-0.48.2-py3-none-manylinux_2_24_x86_64.whl'
INSTALL_DIR = '/kaggle/tmp/lib'
os.makedirs(INSTALL_DIR, exist_ok=True)

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install',
    BNB_WHL_PATH, '--target', INSTALL_DIR, '--no-deps', '--no-index'
])

sys.path.insert(0, INSTALL_DIR)
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

import bitsandbytes as bnb
print(f"âœ… Bitsandbytes: {bnb.__version__}")


import torch
import pandas as pd
import json
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ==================== é…�ç½®å�€ ====================

# Base æ¨¡å�‹è·¯å¾‘
BASE_MODELS = {
    'gemma': '/kaggle/input/gemma/transformers/7b-it/3',
    'gemma2': '/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2',
    'qwen': '/kaggle/input/qwen2.5/transformers/7b-instruct/1',
    'llama32': '/kaggle/input/llama-3.2/transformers/3b-instruct/1',
}

# LoRA adapters è·¯å¾‘
KAGGLE_ADAPTERS = {
    'gemma': '/kaggle/input/llm-prompt-recovery-adapters/gemma-7b_20251204_005138/final_model',
    # 'gemma2': '/kaggle/input/llm-prompt-recovery-adapters/gemma2-7b_20251203_022244/final_model',
    'qwen': '/kaggle/input/llm-prompt-recovery-adapters/qwen-7b_20251204_052459/final_model',
    'llama32': '/kaggle/input/llm-prompt-recovery-adapters/llama32-7b_20251203_054555/final_model',
}

# æ¸¬è©¦æª”æ¡ˆ
TEST_PATH = '/kaggle/input/llm-prompt-recovery/test.csv'

# Magic suffix
MAGIC_SUFFIX = " 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarea"
FALLBACK_PROMPT = "Transform the following text in a more vivid and descriptive way, while maintaining the original meaning and tone."

# ç”¨æ–¼ç‰¹æ®Šå¾Œè™•ç�†çš„å¾Œç¶´ (é‡�å°� "Improve..." prime)
SPECIAL_SUFFIX_FOR_IMPROVE = " with maintaining the original meaning but altering the tone."


# ============ ğŸš€ å„ªåŒ–å�ƒæ•¸ ============
BATCH_SIZE = 6  # æ‰¹æ¬¡å¤§å°� (æ ¹æ“š GPU è¨˜æ†¶é«”èª¿æ•´: 2/4/8)
MAX_LEN = 512 # å¾� 512 é™�åˆ° 384 (æ›´å¿«) (è¼¸å…¥é•·åº¦é™�åˆ¶)
MAX_NEW_TOKENS = 128 # å¾� 128 é™�åˆ° 64 (prompt é€šå¸¸å¾ˆçŸ­) (è¼¸å‡ºé•·åº¦é™�åˆ¶)
USE_8BIT = False  # True = 8-bit é‡�åŒ– (æ›´å¿«ä½†ç²¾åº¦ç¨�é™�)

# å¿«é€Ÿæ¸¬è©¦æ¨¡å¼� (å�ªè·‘å‰� N ç­†)
DEBUG_MODE = False
DEBUG_SAMPLES = 10

print(f"âœ… é…�ç½®å®Œæˆ�")
print(f"ğŸ“� æ¸¬è©¦æª”æ¡ˆ: {TEST_PATH}")
print(f"ğŸ”§ ä½¿ç”¨æ¨¡å�‹: {list(KAGGLE_ADAPTERS.keys())}")
print(f"âš¡ Batch Size: {BATCH_SIZE}, Max Length: {MAX_LEN}")
print(f"ğŸ�› Debug Mode: {DEBUG_MODE}")


test = pd.read_csv(TEST_PATH)

if DEBUG_MODE:
    test = test.head(DEBUG_SAMPLES)
    print(f"âš ï¸� Debug æ¨¡å¼�: å�ªè™•ç�†å‰� {len(test)} ç­†")

print(f"ğŸ“Š æ¸¬è©¦è³‡æ–™: {len(test)} ç­†")
test.head()


def load_model(base_model, adapter_path=None, use_8bit=False):
    """è¼‰å…¥æ¨¡å�‹ - æ”¯æ�´ 4-bit/8-bit é‡�åŒ–"""
    print(f"ğŸ“¥ è¼‰å…¥: {base_model.split('/')[-2]}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # æ‰¹æ¬¡æ�¨è«–å¿…é ˆè¨­å®š
    
    if use_8bit:
        print("âš¡ 8-bit é‡�åŒ– (æ›´å¿«)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        print("âš¡ 4-bit é‡�åŒ–")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"ğŸ”§ è¼‰å…¥ LoRA: {adapter_path.split('/')[-2]}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    print("âœ… å®Œæˆ�")
    return model, tokenizer


def batch_predict_gemma(model, tokenizer, test_df, prime="", batch_size=4, max_len=384):
    """
    Gemma æ‰¹æ¬¡æ�¨è«– - é—œé�µå„ªåŒ–!
    """
    predictions = []
    
    # é �è™•ç�†æ‰€æœ‰æ¨£æœ¬
    prompts = []
    special_indices = []  # è¨˜éŒ„ç‰¹æ®Šæƒ…æ³� (å�Ÿæ–‡=æ”¹å¯«æ–‡)
    
    for idx, row in test_df.iterrows():
        if row.original_text == row.rewritten_text:
            special_indices.append(idx)
            prompts.append(None)
            continue
        
        ot = " ".join(str(row.original_text).split(" ")[:max_len])
        rt = " ".join(str(row.rewritten_text).split(" ")[:max_len])
        
        user_content = f"Find the orginal prompt that transformed original text to new text.\n\nOriginal text: {ot}\n====\nNew text: {rt}"
        conversation = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompt += f"<start_of_turn>model\n{prime}"
        prompts.append(prompt)
    
    # æ‰¹æ¬¡æ�¨è«–
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Gemma [{prime[:15]}...]"):
            batch_prompts = prompts[i:i+batch_size]
            
            # è™•ç�†ç‰¹æ®Šæƒ…æ³�
            valid_prompts = [p for p in batch_prompts if p is not None]
            if not valid_prompts:
                predictions.extend(["Correct grammatical errors in this text."] * len(batch_prompts))
                continue
            
            # Tokenize batch
            inputs = tokenizer(
                valid_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536
            ).to(model.device)
            
            # ç”Ÿæˆ�
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # è§£æ��è¼¸å‡º
            for j, prompt in enumerate(batch_prompts):
                if prompt is None:
                    predictions.append("Correct grammatical errors in this text.")
                    continue
                
                try:
                    result = tokenizer.decode(outputs[j])
                    result = result.split("<start_of_turn>model")[1].split("<end_of_turn>")[0]
                    result = result.replace("<end_of_turn>\n<eos>", "").replace("<end_of_turn>", "")
                    result = result.replace("<start_of_turn>", "").replace("<eos>", "").replace("<bos>", "")
                    result = result.strip().replace('"', '').strip()
                    
                    # å¾Œè™•ç�†
                    result = result.replace("Can you make this", "Make this")
                    result = result.replace("?", ".")
                    result = result.replace("Revise", "Rewrite")
                    result = result.split(":", 1)[-1].strip()
                    
                    if "useruser" in result:
                        result = result.replace("user", "")
                    
                    if result and result[-1].isalnum():
                        result += "."
                    elif result and not result[-1] == ".":
                        result = result[:-1] + "."
                    
                    # âš ï¸� åŠ å…¥ Title Case é��æ¿¾ (ç¬¬ä¸€å��çš„é—œé�µå¾Œè™•ç�†)
                    if len(result.split()) < 100 and len(result.split()) > 2 and "\n" not in result:
                        # æª¢æŸ¥æ˜¯å�¦æœ‰ä¸�ç•¶çš„ Title Case è©�å½™ï¼ˆå�¯èƒ½æ˜¯æ¨¡å�‹å¹»è¦ºï¼‰
                        has_title_case = False
                        for word in result.split()[1:]:
                            if word.istitle() and len(word) > 1:
                                has_title_case = True
                                break
                        if has_title_case:
                            predictions.append(FALLBACK_PROMPT)
                        else:
                            predictions.append(result)
                    else:
                        predictions.append(FALLBACK_PROMPT)
                except:
                    predictions.append(FALLBACK_PROMPT)
    
    return predictions


def batch_predict_qwen(model, tokenizer, test_df, prime="", batch_size=4, max_len=384):
    """Qwen æ‰¹æ¬¡æ�¨è«–"""
    predictions = []
    prompts = []
    
    for idx, row in test_df.iterrows():
        if row.original_text == row.rewritten_text:
            prompts.append(None)
            continue
        
        ot = " ".join(str(row.original_text).split(" ")[:max_len])
        rt = " ".join(str(row.rewritten_text).split(" ")[:max_len])
        
        user_content = f"Find the original prompt that transformed original text to new text.\n\nOriginal text: {ot}\n====\nNew text: {rt}"
        conversation = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt += prime
        prompts.append(prompt)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Qwen [{prime[:15]}...]"):
            batch_prompts = prompts[i:i+batch_size]
            valid_prompts = [p for p in batch_prompts if p is not None]
            
            if not valid_prompts:
                predictions.extend(["Correct grammatical errors in this text."] * len(batch_prompts))
                continue
            
            inputs = tokenizer(valid_prompts, return_tensors="pt", padding=True, 
                             truncation=True, max_length=1536).to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, 
                                   do_sample=False, num_beams=1,
                                   pad_token_id=tokenizer.eos_token_id)
            
            for j, prompt in enumerate(batch_prompts):
                if prompt is None:
                    predictions.append("Correct grammatical errors in this text.")
                    continue
                
                try:
                    result = tokenizer.decode(outputs[j], skip_special_tokens=False)
                    if "<|im_start|>assistant" in result:
                        result = result.split("<|im_start|>assistant")[-1]
                        result = result.split("<|im_end|>")[0]
                    result = result.replace("<|im_start|>", "").replace("<|im_end|>", "")
                    result = result.strip().replace('"', '').strip()
                    
                    result = result.replace("Can you make this", "Make this")
                    result = result.replace("?", ".")
                    result = result.split(":", 1)[-1].strip()
                    
                    if result and result[-1].isalnum():
                        result += "."
                    elif result and not result[-1] == ".":
                        result = result[:-1] + "."
                    
                    if len(result.split()) < 100 and len(result.split()) > 2 and "\n" not in result:
                        predictions.append(result)
                    else:
                        predictions.append(FALLBACK_PROMPT)
                except:
                    predictions.append(FALLBACK_PROMPT)
    
    return predictions


def batch_predict_llama(model, tokenizer, test_df, prime="", batch_size=4, max_len=384):
    """Llama æ‰¹æ¬¡æ�¨è«–"""
    predictions = []
    prompts = []
    
    for idx, row in test_df.iterrows():
        if row.original_text == row.rewritten_text:
            prompts.append(None)
            continue
        
        ot = " ".join(str(row.original_text).split(" ")[:max_len])
        rt = " ".join(str(row.rewritten_text).split(" ")[:max_len])
        
        user_content = f"Find the original prompt that transformed original text to new text.\n\nOriginal text: {ot}\n====\nNew text: {rt}"
        conversation = [{"role": "user", "content": user_content}]
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt += prime
        prompts.append(prompt)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Llama [{prime[:15]}...]"):
            batch_prompts = prompts[i:i+batch_size]
            valid_prompts = [p for p in batch_prompts if p is not None]
            
            if not valid_prompts:
                predictions.extend(["Correct grammatical errors in this text."] * len(batch_prompts))
                continue
            
            inputs = tokenizer(valid_prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=1536).to(model.device)
            
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False, num_beams=1,
                                   pad_token_id=tokenizer.eos_token_id)
            
            for j, prompt in enumerate(batch_prompts):
                if prompt is None:
                    predictions.append("Correct grammatical errors in this text.")
                    continue
                
                try:
                    result = tokenizer.decode(outputs[j], skip_special_tokens=False)
                    if "<|start_header_id|>assistant<|end_header_id|>" in result:
                        result = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                        result = result.split("<|eot_id|>")[0]
                    for token in ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]:
                        result = result.replace(token, "")
                    result = result.strip().replace('"', '').strip()
                    
                    result = result.replace("Can you make this", "Make this")
                    result = result.replace("?", ".")
                    result = result.split(":", 1)[-1].strip()
                    
                    if result and result[-1].isalnum():
                        result += "."
                    elif result and not result[-1] == ".":
                        result = result[:-1] + "."
                    
                    if len(result.split()) < 100 and len(result.split()) > 2 and "\n" not in result:
                        predictions.append(result)
                    else:
                        predictions.append(FALLBACK_PROMPT)
                except:
                    predictions.append(FALLBACK_PROMPT)
    
    return predictions


print("âœ… æ‰¹æ¬¡æ�¨è«–å‡½æ•¸å®šç¾©å®Œæˆ�")


import time

all_predictions = []
start_time = time.time()

# ==================== Gemma ====================
if 'gemma' in KAGGLE_ADAPTERS:
    print("\n" + "="*80)
    print("ğŸš€ Gemma-7B (Prime: Improve... + ç‰¹æ®Šå¾Œç¶´)")
    print("="*80)
    
    model, tokenizer = load_model(
        BASE_MODELS['gemma'], 
        KAGGLE_ADAPTERS['gemma'],
        use_8bit=USE_8BIT
    )
    
    # å�ªç”¨æœ€å¥½çš„ prime (çœ�ä¸€å�Šæ™‚é–“)
    prime = "General prompt: Improve this text using the writing style"
    preds = batch_predict_gemma(model, tokenizer, test, prime=prime, 
                               batch_size=BATCH_SIZE, max_len=MAX_LEN)
    
    # âš ï¸� é—œé�µï¼šé‡�å°�é€™å€‹ Prime çš„ç‰¹æ®Šå¾Œè™•ç�†
    fixed_preds = []
    for p in preds:
        # ç§»é™¤æœ«å°¾å�¥è™Ÿä¸¦åŠ ä¸Šç‰¹å®šå¾Œç¶´
        if p.endswith('.'):
            p = p[:-1]
        p += SPECIAL_SUFFIX_FOR_IMPROVE
        fixed_preds.append(p)
    
    all_predictions.append(fixed_preds)
    print(f"  âœ“ å®Œæˆ�: {len(preds)} ç­†")
    
    del model, tokenizer
    torch.cuda.empty_cache()

# ==================== Gemma2 ====================
if 'gemma2' in KAGGLE_ADAPTERS:
    print("\n" + "="*80)
    print("ğŸš€ Gemma2-9B (Prime: Alter)")
    print("="*80)
    
    model, tokenizer = load_model(
        BASE_MODELS['gemma2'], 
        KAGGLE_ADAPTERS['gemma2'],
        use_8bit=USE_8BIT
    )
    
    prime = "General prompt: Alter"
    preds = batch_predict_gemma(model, tokenizer, test, prime=prime,
                               batch_size=4, max_len=MAX_LEN)
    all_predictions.append(preds)
    print(f"  âœ“ å®Œæˆ�: {len(preds)} ç­†")
    
    del model, tokenizer
    torch.cuda.empty_cache()

# ==================== Qwen ====================
if 'qwen' in KAGGLE_ADAPTERS:
    print("\n" + "="*80)
    print("ğŸš€ Qwen2.5-7B (Prime: Make this text)")
    print("="*80)
    
    model, tokenizer = load_model(
        BASE_MODELS['qwen'], 
        KAGGLE_ADAPTERS['qwen'],
        use_8bit=USE_8BIT
    )
    
    # Prime 3: æ¨¡ä»¿ç¬¬ä¸€å��çš„ pred3.json (ä¸�è¦�è·Ÿå…¶ä»–æ¨¡å�‹é‡�è¤‡)
    prime = "The prompt is: Make this text"
    preds = batch_predict_qwen(model, tokenizer, test, prime=prime,
                              batch_size=BATCH_SIZE, max_len=MAX_LEN)
    all_predictions.append(preds)
    print(f"  âœ“ å®Œæˆ�: {len(preds)} ç­†")
    
    del model, tokenizer
    torch.cuda.empty_cache()

# ==================== Llama ====================
if 'llama32' in KAGGLE_ADAPTERS:
    print("\n" + "="*80)
    print("ğŸš€ Llama-3.2-3B (Prime: Rewrite)")
    print("="*80)
    
    model, tokenizer = load_model(
        BASE_MODELS['llama32'], 
        KAGGLE_ADAPTERS['llama32'],
        use_8bit=USE_8BIT
    )
    
    prime = "The prompt is: Rewrite"
    preds = batch_predict_llama(model, tokenizer, test, prime=prime,
                               batch_size=BATCH_SIZE, max_len=MAX_LEN)
    all_predictions.append(preds)
    print(f"  âœ“ å®Œæˆ�: {len(preds)} ç­†")
    
    del model, tokenizer
    torch.cuda.empty_cache()

elapsed = time.time() - start_time
print(f"\nâ�±ï¸� ç¸½æ�¨è«–æ™‚é–“: {elapsed/60:.1f} åˆ†é�˜")
print(f"ğŸ“Š æ”¶é›† {len(all_predictions)} çµ„é �æ¸¬")


if len(all_predictions) > 0:
    print(f"ğŸ”„ Ensemble {len(all_predictions)} çµ„é �æ¸¬...")
    final_predictions = [' '.join(preds) for preds in zip(*all_predictions)]
    final_predictions = [pred + MAGIC_SUFFIX for pred in final_predictions]
else:
    print("âš ï¸� ä½¿ç”¨ fallback")
    final_predictions = [FALLBACK_PROMPT + MAGIC_SUFFIX] * len(test)

print(f"âœ… æœ€çµ‚é �æ¸¬: {len(final_predictions)} ç­†")
print(f"\nğŸ“‹ å‰� 3 ç­†:")
for i, pred in enumerate(final_predictions[:3]):
    print(f"  [{i}] {pred[:100]}...")


submission = pd.read_csv('/kaggle/input/llm-prompt-recovery/sample_submission.csv')
submission['rewrite_prompt'] = final_predictions
submission.to_csv('submission.csv', index=False)

print("âœ… å·²ä¿�å­˜: submission.csv")
print(f"ğŸ“Š å…± {len(submission)} ç­†")

submission.head()

