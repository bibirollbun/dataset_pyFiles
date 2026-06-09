# =============================================================================
# 實驗 C-9 - Step 2：執行預測（修正版：4bit 失敗自動 fallback FP16 + 修多卡輸入 device）
# =============================================================================

# 確保已經執行 Step 1 安裝 bitsandbytes
!pip install /kaggle/input/bitsandbytes-package/bitsandbytes-0.49.0-py3-none-manylinux_2_24_x86_64.whl --force-reinstall

import os
import gc
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 基本環境設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU 數量: {torch.cuda.device_count()}")

# 驗證 bitsandbytes
try:
    import bitsandbytes
    print(f"✓ bitsandbytes: {bitsandbytes.__version__}")
except Exception as e:
    print("❌ bitsandbytes 未安裝！請先執行 Step 1")
    raise

# ============================================================
# 載入資料
# ============================================================
data_path = Path('/kaggle/input/llm-prompt-recovery')

try:
    test = pd.read_csv(data_path / 'test.csv')
    test = test.fillna("")
    print(f"測試資料: {len(test)} 筆")
except Exception as e:
    print(f"載入錯誤: {e}")
    raise

# ============================================================
# 載入模型（修正版：先試 4bit，失敗自動 fallback FP16）
# ============================================================
MODEL_PATH = "/kaggle/input/gemma/transformers/7b-it/3"
if not os.path.exists(MODEL_PATH):
    for p in ["/kaggle/input/gemma/transformers/7b-it/2",
              "/kaggle/input/gemma/transformers/7b-it/1"]:
        if os.path.exists(p):
            MODEL_PATH = p
            break

print(f"模型: {MODEL_PATH}")

print("載入 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

def load_model_try_4bit_then_fp16():
    # 多卡 + CPU buffer，避免 OOM
    max_memory = {0: "14GB", 1: "14GB", "cpu": "30GB"}

    # 先試 4bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        print("載入模型（嘗試 4bit bitsandbytes）...")
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        print("✓ 4bit 載入成功")
        return m

    except Exception as e:
        print("⚠️ 4bit 載入失敗，原因如下（將自動改用 FP16 多卡切分）:")
        print(e)

        # 釋放殘留 GPU 記憶體
        gc.collect()
        torch.cuda.empty_cache()

        print("載入模型（FP16 / device_map=auto）...")
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        print("✓ FP16 載入成功（fallback）")
        return m

model = load_model_try_4bit_then_fp16()
model.eval()
print("✓ 模型載入完成")

# ============================================================
# 改進的 4 個範例
# ============================================================
EXAMPLES = [
    {"original": "Hey, this thing is broken and needs fixing ASAP!",
     "rewritten": "This item requires immediate attention and repair.",
     "prompt": "Rewrite in a formal, professional tone"},

    {"original": "The implementation necessitates comprehensive analysis of multifaceted parameters.",
     "rewritten": "We need to carefully look at all the different factors before we implement this.",
     "prompt": "Simplify and make more conversational"},

    {"original": "First, heat the oven. Then, mix ingredients. After that, pour into pan. Finally, bake for 30 minutes.",
     "rewritten": "Instructions:\n1. Heat the oven\n2. Mix ingredients\n3. Pour into pan\n4. Bake for 30 minutes",
     "prompt": "Convert to a numbered list format"},

    {"original": "Due to the fact that we have been experiencing some technical difficulties and issues with the system, we have made the decision to postpone the launch until further notice.",
     "rewritten": "We're postponing the launch due to technical issues.",
     "prompt": "Make more concise and remove redundancy"},
]

DEFAULT_PROMPT = "Rewrite this text in a different style."

# ============================================================
# 改進的 Prompt 格式
# ============================================================
def build_prompt(original, rewritten):
    prompt = (
        "You are a prompt predictor. Given an original text and its rewritten version, "
        "predict the exact prompt that was used. Output ONLY the prompt.\n\n"
    )

    for i, ex in enumerate(EXAMPLES, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Original: {ex['original']}\n"
        prompt += f"Rewritten: {ex['rewritten']}\n"
        prompt += f"Prompt: {ex['prompt']}\n\n"

    prompt += f"Original: {str(original)[:850]}\n"
    prompt += f"Rewritten: {str(rewritten)[:850]}\n"
    prompt += f"Prompt:"
    return prompt

# ============================================================
# generate（修正版：支援 device_map=auto 多卡，不要硬塞 cuda:0）
# ============================================================
def _get_first_cuda_device(model):
    # device_map="auto" 時，輸入必須丟到「模型第一個 CUDA device」
    if hasattr(model, "hf_device_map"):
        devices = []
        for d in model.hf_device_map.values():
            if isinstance(d, str) and d.startswith("cuda"):
                devices.append(d)
        if len(devices) > 0:
            return devices[0]
    # fallback：如果沒 hf_device_map，就用參數所在 device
    return str(next(model.parameters()).device)

FIRST_DEVICE = _get_first_cuda_device(model)
print("模型第一張卡:", FIRST_DEVICE)

def generate(prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(FIRST_DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        response = response.strip().split("\n")[0].strip()

        for p in ["Example", "**Answer:**", "Answer:", "Prompt:", "prompt:", "Instruction:", "**"]:
            if response.startswith(p):
                response = response[len(p):].strip()

        result = response.strip('"').strip("'")[:250]
        if result and len(result) > 3:
            return result
        return DEFAULT_PROMPT

    except Exception as e:
        print(f"Generate error: {e}")
        return DEFAULT_PROMPT

# ============================================================
# 執行預測
# ============================================================
print(f"\n開始預測 {len(test)} 筆...")
print("實驗: C-9 Optimized Fixed 4-shot")
print("改進: 更好的範例 + 更清晰格式 + 更長生成 + 4bit fallback + 多卡輸入修正")

results = []

for idx, row in tqdm(test.iterrows(), total=len(test)):
    try:
        original = str(row.get('original_text', ''))
        rewritten = str(row.get('rewritten_text', ''))
        row_id = row.get('id', idx)

        pred = generate(build_prompt(original, rewritten))
        results.append({'id': row_id, 'rewrite_prompt': pred})

    except Exception as e:
        print(f"Error at row {idx}: {e}")
        results.append({'id': row.get('id', idx), 'rewrite_prompt': DEFAULT_PROMPT})

    if (idx + 1) % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{idx+1}/{len(test)}] 已處理")

# ============================================================
# 產生 submission
# ============================================================
submission = pd.DataFrame(results)
submission['rewrite_prompt'] = submission['rewrite_prompt'].fillna(DEFAULT_PROMPT)
submission['rewrite_prompt'] = submission['rewrite_prompt'].replace('', DEFAULT_PROMPT)
submission = submission[['id', 'rewrite_prompt']]
submission.to_csv("submission.csv", index=False)

print(f"\n✓ 完成！")
print(f"submission.csv: {len(submission)} 筆")
print(submission.head())


