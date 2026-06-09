%%writefile constraints.txt

bitsandbytes==0.48.2
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvrtc-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-curand-cu12==10.3.5.147
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nvjitlink-cu12==12.4.127
scikit-learn==1.7.2


%pip install --no-index \
  --find-links=/kaggle/input/offline-packages \
  -c constraints.txt bitsandbytes hdbscan umap_learn peft


%pip install --force-reinstall --no-index /kaggle/input/protobuf/protobuf-4.25.3-cp37-abi3-manylinux2014_x86_64.whl


!cp ../input/llm-prompt-recovery/test.csv .


import sys
import os
import subprocess
import pandas as pd
import numpy as np
import torch
import json
import gc
import re
import warnings
import logging
from tqdm import tqdm
from typing import List, Tuple, Dict

# Transformers & PEFT
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import T5Tokenizer, T5EncoderModel
from peft import PeftModel
import hdbscan
import umap

# ---------------- 設定區塊 ----------------
# [Magic String]
MAGIC_PREFIX = " 'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea"
DEFAULT_PROMPT = 'Rewrite the text.'

# [Paths] 請修改為您的實際路徑
# 注意：這裡保留了原始的模型路徑設定，您需要確保這些路徑在您的執行環境中是有效的
BASE_MODEL_PATH = "/kaggle/input/llama-3.1/transformers/8b-instruct/2"
LORA_ADAPTER_DIR = "/kaggle/input/qlora-out-llama31-prompt-recovery/qlora_out_llama31_prompt_recovery"
T5_MODEL_PATH = "/kaggle/input/t5/transformers/base/1" # 建議使用 HF 版本

# [Modified] 預設輸入路徑改為當前目錄下的 test.csv
TEST_DATA_PATH = "test.csv" 

# ---------------- Log 抑制 ----------------
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ==========================================
# 2. 輔助函數：數據清洗 (From data_process)
# ==========================================
import unicodedata

def clean_text_data(df, text_cols=['original_text', 'rewritten_text'], max_len=1000):
    """
    強化版數據清洗函數
    Args:
        df: Pandas DataFrame
        text_cols: 需要清洗的欄位列表
        max_len: 每個文本允許的最大字符長度 (防止 OOM)
    """
    print(f"Cleaning text data (Robust Mode)... processing {len(df)} rows")
    
    for col in text_cols:
        if col not in df.columns:
            continue

        # 1. [Critical] 處理 NaN 和非字串類型，防止 AttributeError
        df[col] = df[col].fillna("").astype(str)

        # 2. [Critical] 長度截斷 (Truncation) - 這是防止 Pipeline 崩潰的第一道防線
        # 如果文本超過 max_len，直接截斷。對於 Prompt Recovery，過長的文本通常不需要完整保留
        df[col] = df[col].str.slice(0, max_len)

        # 3. 基礎 HTML 解碼與符號標準化
        df[col] = df[col].str.replace('&#x200B;', '', regex=False)
        df[col] = df[col].str.replace('&gt;', '>', regex=False)
        df[col] = df[col].str.replace('&lt;', '<', regex=False)
        df[col] = df[col].str.replace('&amp;', '&', regex=False)
        df[col] = df[col].str.replace('&nbsp;', ' ', regex=False)
        
        # 智能引號轉為標準 ASCII 引號
        df[col] = df[col].str.replace('’', "'", regex=False)
        df[col] = df[col].str.replace('‘', "'", regex=False)
        df[col] = df[col].str.replace('“', '"', regex=False)
        df[col] = df[col].str.replace('”', '"', regex=False)

        # 4. [Critical] 移除不可見字符與控制字符 (Control Characters)
        # 這裡使用 regex 移除 ASCII 0-31 (除了換行符號先保留，稍後處理)
        # 許多 CSV 錯誤是由奇怪的 binary characters 造成的
        df[col] = df[col].str.replace(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', regex=True)

        # 5. 空白標準化 (Whitespace Normalization)
        # 將 \n, \t, \r 和多重空白全部轉為單一空格
        # 這能避免 Regex 在處理大量空白時發生效能問題
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

    # 針對 rewritten_text 的特定清洗 (移除 LLM 常見廢話)
    if 'rewritten_text' in df.columns:
        col = 'rewritten_text'
        
        # 移除 Markdown 粗體、標題符號
        df[col] = df[col].str.replace(r'\*\*', '', regex=True)
        df[col] = df[col].str.replace(r'##', '', regex=True)
        
        # 移除常見 LLM 開場白 (Case insensitive)
        # 加入了更多常見變體，如 "Certainly", "Okay"
        patterns_to_remove = [
            r"(?i)^sure[,:\s]*",
            r"(?i)^here['’`]?s[,:\s]*",
            r"(?i)^certainly[,:\s]*",
            r"(?i)^okay[,:\s]*",
            r"(?i)^ok[,:\s]*",
            r"(?i)^as an ai[,:\s]*",
            r"^\*\*.*\*\*[:,\s]*", # **Title**: ...
            r"^\<.*\>[:,\s]*"      # <Tag>: ...
        ]
        
        for pat in patterns_to_remove:
            df[col] = df[col].str.replace(pat, '', regex=True)

    # 6. [Critical] 最終清洗與空值防護
    for col in text_cols:
        df[col] = df[col].str.strip()
        
        # 如果清洗後變成空字串，填入一個 safe placeholder
        # 這是為了防止後續 Tokenizer 遇到 empty string 報錯
        df.loc[df[col] == "", col] = "Empty text."

    return df

# ==========================================
# 3. 核心邏輯：標籤生成與聚類 (From data_process)
# ==========================================

def validate_and_parse_tags(tags_str: str) -> Tuple[bool, List[str]]:
    """驗證 Tags 輸出格式"""
    if not tags_str: return False, []
    cleaned_str = tags_str.strip().split('\n')[0]
    cleaned_str = cleaned_str.replace(' ,', ',').replace(', ', ',').replace(' ,', ',')
    
    if ',' not in cleaned_str: 
        if ' ' not in cleaned_str and len(cleaned_str) > 0: 
             return True, [cleaned_str.lower()]
        return False, []

    if not re.match(r'^[a-zA-Z0-9\- ,]+$', cleaned_str, re.IGNORECASE):
         return False, []

    tags_list = [tag.strip().lower() for tag in cleaned_str.split(',') if tag.strip()]
    return (True, tags_list) if tags_list else (False, [])

def batch_generate_tags(model, tokenizer, sentences: List[str], batch_size: int = 30) -> List[List[str]]:
    """
    使用 Llama 生成標籤 (Stage 1 & 2)
    已恢復「錯誤反饋重試」機制 (Error Feedback Retry Logic)
    [Modified] 加入 tqdm 進度條
    """
    n_total = len(sentences)
    final_results = [None] * n_total
    pending_indices = list(range(n_total))
    
    # 用來記錄失敗的輸出，以便在下一次 prompt 中以此作為反面教材
    wrong_outputs_map: Dict[int, str] = {} 

    # --- Template Definition ---
    initial_prompt_template = """
    You are a tagging assistant for text classification.

    Task:
    Given one sentence, output 10 concise tags that best describe:
    - content/topic (e.g., poem, code, news, legal)
    - style/tone (e.g., humorous, formal, sarcastic)
    - structure/formatting (e.g., dialogue, list, markdown)

    Rules:
    - Output only a comma-separated list of tags.
    - Use singular, lowercase English words.
    - Do NOT invent meta-comments, only tags.

    Example 1:
    Sentence: "Rewrite this poem into a more modern style, but keep the original rhyme scheme."
    Tags: poem, rewrite, modern, rhyme, style

    Example 2:
    Sentence: "Summarize the following technical blog post about neural networks in bullet points."
    Tags: summary, technical, neural-networks, bullet-list, instruction

    Now process this input.
    Sentence: "{sentence}"
    Tags:
    """.strip()
    
    # 建立一個通用的指令區塊，用於錯誤修正 Prompt
    generic_instruction_block = initial_prompt_template.replace('"{sentence}"', '[SENTENCE]').replace('Tags:', '[TAGS OUTPUT LOCATION]')

    max_retries = 5
    
    for attempt in range(max_retries + 1):
        if not pending_indices: break
        
        # [Modified] 使用 tqdm 顯示當前 Attempt 的進度
        # leave=False 表示當該次嘗試結束後，進度條會消失或被覆蓋，保持畫面整潔
        with tqdm(total=len(pending_indices), desc=f"  - Attempt {attempt}/{max_retries}", leave=False) as pbar:
            
            # --- Batch Loop ---
            for i in range(0, len(pending_indices), batch_size):
                batch_indices = pending_indices[i : i + batch_size]
                batch_sentences = [sentences[idx] for idx in batch_indices]
                
                # 1. 動態構建 Prompt (加入錯誤反饋邏輯)
                batch_prompts = []
                for idx, sent in zip(batch_indices, batch_sentences):
                    
                    # 如果是重試且之前有錯誤記錄，使用「糾錯 Prompt」
                    if attempt > 0 and idx in wrong_outputs_map:
                        prev_bad_output = wrong_outputs_map[idx]
                        
                        prompt = f"""
                        Your previous output for the sentence: "{sent}" was: "{prev_bad_output}".
                        This output was INCORRECTLY FORMATTED. 
                        STRICTLY follow the Rules: Output ONLY a comma-separated list of lowercase tags. NO other text, explanation, or punctuation and no quotes.

                        Original Instructions:
                        {generic_instruction_block}
                        
                        Now, process the input again.
                        Sentence: "{sent}"
                        Tags:
                        """.strip()
                    else:
                        # 第一次嘗試或無錯誤記錄，使用「標準 Prompt」
                        prompt = initial_prompt_template.replace("{sentence}", sent)
                    
                    batch_prompts.append(prompt)
                
                # 2. 模型推論
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
                
                with torch.no_grad():
                    # 使用 context manager 暫時禁用 LoRA (使用 Base Model 能力)
                    with model.disable_adapter(): 
                        outputs = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=64, 
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.7,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                
                generated_ids = outputs[:, inputs.input_ids.shape[-1]:]
                responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # 3. 驗證與結果處理
                next_pending = []
                for idx, response in zip(batch_indices, responses):
                    # 基本清理
                    clean_response = response.split('\n')[0].strip()
                    
                    is_valid, tags = validate_and_parse_tags(clean_response)
                    
                    if is_valid:
                        final_results[idx] = tags
                        # 成功後移除錯誤記錄
                        if idx in wrong_outputs_map: del wrong_outputs_map[idx]
                    else:
                        # 驗證失敗：記錄錯誤輸出，並加入下一輪待處理清單
                        wrong_outputs_map[idx] = clean_response
                        
                        if attempt == max_retries:
                            # 最後一次嘗試仍失敗，給空 list
                            final_results[idx] = []
                        else:
                            next_pending.append(idx)
                
                # [Modified] 更新進度條
                pbar.update(len(batch_indices))

        # 更新下一輪的待處理清單
        pending_indices = next_pending
        if pending_indices and attempt < max_retries:
            print(f"  > Retrying {len(pending_indices)} samples with error feedback...")

    return [res if res is not None else [] for res in final_results]

def safe_postprocess_prediction(
    text: str,
    magic: str,
    min_words: int = 2,
    max_words: int = 300,
    extra_tail: str = "",
):
    """
    防錯處理函式：
    - text: 模型生成的原始 prompt
    - magic: 驗證失敗時的回退字串 (Fallback)
    - extra_tail: 成功時要附加的後綴 (即 MAGIC_PREFIX)
    """
    try:
        # 1. 基礎型別與空值檢查
        if not isinstance(text, str) or len(text.strip()) == 0:
            return magic

        # 2. 清理雜訊
        x = text.strip().replace('"', "").strip()
        
        # 清掉 useruser 類型的異常輸出 (常見於 Llama 訓練過擬合)
        if "useruser" in x:
            x = x.replace("user", "")

        x = x.strip()
        if len(x) == 0:
            return magic

        # 3. 結尾標點符號標準化
        last_char = x[-1]
        if last_char.isalnum():
            x = x + "."
        else:
            if last_char not in [".", "!", "?"]:
                x = x[:-1] + "."
            else:
                x = x[:-1] + "." # 保持原樣或強制統一 (此處邏輯為保持有點)

        # 4. 加上後綴 (注意：先加後綴再檢查長度，或者先檢查本文再加後綴)
        # 這裡的邏輯是：如果本文合法，加上後綴返回。
        # 如果本文不合法，返回 magic (magic 應該本身就包含後綴)
        final_candidate = x + extra_tail

        # 5. 長度與格式檢查
        # 檢查 "本文部分" (不含 tail) 的字數，避免模型崩潰輸出數千字
        n_words = len(x.split())
        
        if n_words < min_words:
            return magic
        
        if n_words > max_words:
            # 如果太長，可能是幻覺，回退到 magic
            return magic

        # 檢查換行 (通常 Prompt 不應該有多行)
        if "\n" in x:
            # 嘗試只取第一行，還是回退？這裡選擇回退以策安全
            return magic

        return final_candidate

    except Exception:
        # 發生任何未預期的錯誤 (Unicode Error 等) → 回退 magic
        return magic

def get_t5_embeddings(t5_model, t5_tokenizer, tags_list: List[str]) -> np.ndarray:
    """計算一組 tags 的平均 embedding"""
    if not tags_list: return None
    inputs = t5_tokenizer(tags_list, return_tensors="pt", padding=True, truncation=True).to(t5_model.device)
    with torch.no_grad():
        outputs = t5_model(**inputs)
    # Mean Pooling
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

def generate_cluster_names(model, tokenizer, cluster_tags_list: List[List[str]]) -> List[str]:
    """為群集命名 (Stage 4)"""
    if not cluster_tags_list: return []
    
    prompts = []
    for tags in cluster_tags_list:
        tags_str = ", ".join(tags)
        prompt = (
        "Summarize the following tags into ONE short topic name.\n"
        "Output ONLY the topic name, no explanation, no extra text.\n\n"
        "Examples:\n"
        "Tags: apple, banana, cherry\n"
        "Topic: Fruit\n\n"
        "Tags: stock, market, finance, crash\n"
        "Topic: Economy\n\n"
        f"Tags: {tags_str}\n"
        "Topic:"
        )
        prompts.append(prompt)
        
    names = []
    batch_size = 30
    
    # [Modified] 加入 tqdm 顯示命名進度
    for i in tqdm(range(0, len(prompts), batch_size), desc="  - Naming Clusters"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            with model.disable_adapter(): # 同樣使用 Base Model 能力
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=8,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    eos_token_id=tokenizer.eos_token_id
                )
        responses = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        for res in responses:
            clean = res.split('\n')[0].strip()
            for sep in [",", "-", "–", "—", ":"]:
                clean = clean.split(sep)[0]
            names.append(clean.strip())
            
    return names

# ==========================================
# 4. 主流程 (MAIN)
# ==========================================

def main():
    print(">>> Starting Submission Pipeline...")

    # --- [Modified] 參數處理 (Argument Parsing) ---
    # 預設使用上方定義的 TEST_DATA_PATH (即 "test.csv")
    input_csv_path = TEST_DATA_PATH
    output_csv_path = "submission.csv"

    print(f"Target Input CSV: {input_csv_path}")
    print(f"Target Output CSV: {output_csv_path}")
    
    # --- A. 載入模型 (Load Models) ---
    print(f"Loading Llama 3.1 + LoRA from {BASE_MODEL_PATH}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 1. Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    llama_tokenizer.pad_token = "<|eot_id|>"
    llama_tokenizer.padding_side = "left"

    # 2. LoRA Adapter (Load it now, use context manager to disable later if needed)
    print(f"Loading LoRA from {LORA_ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()

    # 3. T5 Model
    print(f"Loading T5 from {T5_MODEL_PATH}...")
    try:
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH, local_files_only=True, legacy=False)
        t5_model = T5EncoderModel.from_pretrained(T5_MODEL_PATH, local_files_only=True, device_map="auto")
        t5_model.eval()
    except Exception as e:
        print(f"Error loading T5: {e}. Tag clustering will be skipped.")
        t5_model = None


    # --- B. 讀取與清洗資料 (Data Prep) ---
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    test_df = pd.read_csv(input_csv_path)
    test_df = clean_text_data(test_df)
    
    print(f"Processing {len(test_df)} rows from {input_csv_path}...")

    # --- C. 標籤生成 (Tag Generation) ---
    # 由於 batch_generate_tags 內部已加入 tqdm，這裡只需 print 標題
    print("\n[Stage 1/4] Generating Tags for Original Text...")
    tags1_list = batch_generate_tags(model, llama_tokenizer, test_df['original_text'].tolist(), batch_size=30)
    
    print("\n[Stage 2/4] Generating Tags for Rewritten Text...")
    tags2_list = batch_generate_tags(model, llama_tokenizer, test_df['rewritten_text'].tolist(), batch_size=30)

    # --- D. 聚類與命名 (Clustering & Naming) ---
    print("\n[Stage 3/4] Clustering & Identification...")
    final_prompts = []
    
    # [Modified] 加入 desc 描述
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  - Processing Rows"):
        t1 = tags1_list[idx]
        t2 = tags2_list[idx]
        common_themes = []

        # 只有在 T5 和 UMAP/HDBSCAN 可用時才執行聚類
        if t5_model and umap and hdbscan and t1 and t2:
            try:
                all_tags = t1 + t2
                if len(all_tags) >= 2:
                    # Embedding
                    vectors = get_t5_embeddings(t5_model, t5_tokenizer, all_tags)
                    
                    # UMAP (Dim Reduction)
                    if len(all_tags) >= 5:
                        n_neighbors = min(5, len(all_tags) - 1)
                        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=min(5, len(all_tags)-2), metric='cosine')
                        vectors = reducer.fit_transform(vectors)
                    else:
                        vectors = vectors.astype(np.float64)
                    
                    # HDBSCAN (Clustering)
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
                    labels = clusterer.fit_predict(vectors)
                    
                    # Grouping
                    source_map = [0]*len(t1) + [1]*len(t2)
                    clusters = {}
                    for tag, lbl, src in zip(all_tags, labels, source_map):
                        if lbl == -1: continue
                        if lbl not in clusters: clusters[lbl] = {'tags': [], 'src': set()}
                        clusters[lbl]['tags'].append(tag)
                        clusters[lbl]['src'].add(src)
                    
                    # 找出同時包含 source 0 (original) 和 source 1 (rewritten) 的群集
                    themes_tags_list = []
                    for lbl, data in clusters.items():
                        if 0 in data['src'] and 1 in data['src']:
                            themes_tags_list.append(list(set(data['tags'])))
                    
                    # 命名這些主題 (generate_cluster_names 內部已經有 tqdm 了)
                    if themes_tags_list:
                        common_themes = generate_cluster_names(model, llama_tokenizer, themes_tags_list)
            except Exception as e:
                pass # 忽略聚類錯誤，退回使用普通 Prompt

        # --- E. 構建 Prompt (Prompt Construction) ---
        original_text = row['original_text']
        rewritten_text = row['rewritten_text']
        
        # 根據是否有 common_themes 加入負面約束
        if common_themes:
            full_negative_list = ", ".join(common_themes)
            user_content = (
                "Given the Original Text and its Rewritten Text JSON, output what was the prompt used to achieve the rewrite."
                f"\n{{\n'Original Text':'{original_text}'"
                f"\n'Rewritten Text':'{rewritten_text}'\n}}"
                '\nMake the prompt short and direct'
                f"\nIt is not about {full_negative_list} related topic or style transformation tasks."
                '\nReturn using the following JSON structure:'
                '{"prompt": "Your best guess for the prompt used"}'
                '\nReturn a valid JSON as output and nothing more.'
            )
        else:
            user_content = (
                "Given the Original Text and its Rewritten Text JSON, output what was the prompt used to achieve the rewrite."
                f"\n{{\n'Original Text':'{original_text}'"
                f"\n'Rewritten Text':'{rewritten_text}'\n}}"
                '\nMake the prompt short and direct'
                '\nReturn using the following JSON structure:'
                '{"prompt": "Your best guess for the prompt used"}'
                '\nReturn a valid JSON as output and nothing more.'
            )
        final_prompts.append(user_content)

    # --- F. 最終推理 (Final Inference) ---
    print("\n[Stage 4/4] Final Prompt Recovery Inference...")
    results = []
    batch_size = 30 
    
    # 建立一個全域的 Fallback 完整字串 (包含後綴)
    FALLBACK_FULL_STRING = DEFAULT_PROMPT + MAGIC_PREFIX

    # [Modified] 加入 desc 描述
    for i in tqdm(range(0, len(test_df), batch_size), desc="  - Recovering Prompts"):
        batch_df_slice = test_df.iloc[i : i + batch_size]
        batch_user_prompts = final_prompts[i : i + batch_size]
        
        formatted_inputs = []
        for p in batch_user_prompts:
            msgs = [{"role": "user", "content": p}]
            formatted_inputs.append(llama_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        
        inputs = llama_tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512, # 限制最大生成 token，防止 OOM
                do_sample=False, 
                pad_token_id=llama_tokenizer.pad_token_id,
                eos_token_id=llama_tokenizer.eos_token_id
            )
            
        generated_ids = outputs[:, inputs.input_ids.shape[-1]:]
        responses = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for idx, (row_idx, row), response in zip(range(len(batch_df_slice)), batch_df_slice.iterrows(), responses):
            # 1. 提取 JSON 內容
            cleaned_prompt = "Rewrite the text." # 初始預設值
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx+1]
                    data = json.loads(json_str)
                    cleaned_prompt = data.get("prompt", response)
                else:
                    # 如果找不到 JSON，做簡單清理
                    cleaned_prompt = response.strip()
                    if "Here is the prompt" in cleaned_prompt:
                        cleaned_prompt = cleaned_prompt.split("\n")[-1]
            except:
                cleaned_prompt = response # JSON 解析失敗時使用原始回應
            
            # 2. 呼叫 safe_postprocess_prediction
            with model.disable_adapter(): 
                final_output = safe_postprocess_prediction(
                    text=cleaned_prompt,
                    magic=FALLBACK_FULL_STRING, 
                    min_words=2,
                    max_words=300,
                    extra_tail=MAGIC_PREFIX
                )
            
            results.append({
                "id": row['id'],
                "rewrite_prompt": final_output
            })

    # --- G. 存檔 ---
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_csv_path, index=False)
    print(f"\nSubmission saved successfully to {output_csv_path}!")

if __name__ == "__main__":
    main()

