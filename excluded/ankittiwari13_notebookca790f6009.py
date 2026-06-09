# !pip install --upgrade transformers
# !pip uninstall -y transformers==4.38.2 accelerate==0.30.1
# !pip install -U transformers accelerate sentencepiece
# !pip install -U transformers==4.38.2
# !pip install transformers==4.40.2 accelerate==0.30.1 -q
# !pip install flash-attn --no-build-isolation
# from transformers.cache_utils import DynamicCache


import torch

# Clears the GPU memory cache
torch.cuda.empty_cache()

import tensorflow as tf
from tensorflow.keras import backend as K

# Clear the TensorFlow session to free up memory
K.clear_session()
import gc

gc.collect()
tf.compat.v1.Session().close()


# ################################################################################
# # ARC 2025 - CPU-ONLY VERSION
# ################################################################################

# import json
# import os
# import numpy as np
# import copy
# from pathlib import Path
# import time
# import re
# import sys
# import traceback

# # --- Essential Imports -------------------------------------------------------
# try:
#     import torch
#     import transformers
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     print(f"Transformers version: {transformers.__version__}")
#     print(f"Torch version: {torch.__version__}")
# except ImportError:
#     print("Transformers or Torch not found. LLM will not work.")
#     AutoModelForCausalLM = None
#     torch = None

# try:
#     from scipy.ndimage import label
#     print("Scipy loaded successfully.")
# except ImportError:
#     print("Scipy not found. Object detection heuristics disabled.")
#     label = None

# # --- Configuration -----------------------------------------------------------
# DATA_DIR = Path("/kaggle/input/arc-prize-2025")
# MODEL_DIR = Path("/kaggle/input/deepseek-coder-v2/transformers/deepseek-coder-v2-lite-base/1")

# TEST_CHALLENGES_FILE = DATA_DIR / "arc-agi_test_challenges.json"
# SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.json"
# SUBMISSION_FILE = "submission.json"

# device = "cpu"

# # --- Grid Validation ---------------------------------------------------------
# def validate_grid(grid, default_grid):
#     try:
#         if not isinstance(grid, list):
#             return default_grid
#         if len(grid) == 0 or len(grid) > 30:
#             return default_grid
#         width = len(grid[0])
#         if width == 0 or width > 30:
#             return default_grid

#         out = []
#         for r_idx, row in enumerate(grid):
#             if not isinstance(row, list) or len(row) != width:
#                 return default_grid
#             new_row = []
#             for c_idx, cell in enumerate(row):
#                 try:
#                     val = int(cell)
#                     if not (0 <= val <= 9):
#                         val = default_grid[r_idx][c_idx] % 10
#                     new_row.append(val)
#                 except Exception:
#                     return default_grid
#             out.append(new_row)
#         return out
#     except Exception:
#         return default_grid

# # --- Heuristic Solvers -------------------------------------------------------
# def get_objects(grid_np):
#     if label is None:
#         return [], 0
#     labeled, n = label(grid_np != 0)
#     objs = []
#     for i in range(1, n + 1):
#         mask = (labeled == i)
#         coords = np.argwhere(mask)
#         colors, counts = np.unique(grid_np[mask], return_counts=True)
#         objs.append({
#             "mask": mask,
#             "coords": coords,
#             "color": int(colors[np.argmax(counts)]),
#             "size": len(coords)
#         })
#     objs.sort(key=lambda o: o["size"], reverse=True)
#     return objs, n

# def _solve_identity(x): return x
# def _solve_flip_h(x): return np.fliplr(x)
# def _solve_flip_v(x): return np.flipud(x)
# def _solve_rotate_90(x): return np.rot90(x, k=1)
# def _solve_rotate_180(x): return np.rot90(x, k=2)
# def _solve_rotate_270(x): return np.rot90(x, k=3)
# def _solve_keep_largest_object(x):
#     objs, _ = get_objects(x)
#     if not objs: return x
#     g = np.zeros_like(x)
#     g[objs[0]["mask"]] = objs[0]["color"]
#     return g
# def _solve_keep_smallest_object(x):
#     objs, _ = get_objects(x)
#     if not objs: return x
#     g = np.zeros_like(x)
#     g[objs[-1]["mask"]] = objs[-1]["color"]
#     return g

# HEURISTIC_SOLVERS = [
#     _solve_identity, _solve_flip_h, _solve_flip_v,
#     _solve_rotate_90, _solve_rotate_180, _solve_rotate_270,
#     _solve_keep_largest_object, _solve_keep_smallest_object,
# ]

# def check_heuristic(solver_func, task):
#     try:
#         for p in task["train"]:
#             inp = np.array(p["input"], dtype=int)
#             out = solver_func(inp).tolist()
#             if out != p["output"]:
#                 return None
#         return lambda x: solver_func(np.array(x, dtype=int)).tolist()
#     except Exception:
#         return None

# # --- LLM Solver --------------------------------------------------------------
# def load_model():
#     """Loads the DeepSeek model and tokenizer for CPU."""
#     if AutoModelForCausalLM is None:
#         print("Cannot load model, transformers library not found.")
#         return None, None
        
#     if not MODEL_DIR.exists():
#         print(f"CRITICAL ERROR: Model directory not found: {MODEL_DIR}")
#         return None, None

#     print(f"Loading model from: {MODEL_DIR}")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

#         # Load model on CPU
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_DIR,
#             torch_dtype=torch.float32,
#             device_map={"": "cpu"},
#             trust_remote_code=True
#         ).to("cpu")

#         # ============================================================
#         # 1ï¸�âƒ£ DynamicCache patch (fix for get_max_length issue)
#         # ============================================================
#         from transformers.cache_utils import DynamicCache
#         if not hasattr(DynamicCache, "get_max_length"):
#             def get_max_length(self):
#                 return self.get_seq_length() if hasattr(self, "get_seq_length") else 0
#             DynamicCache.get_max_length = get_max_length
#             print("âœ… Patched DynamicCache.get_max_length() for new Transformers API")

#         # ============================================================
#         # 2ï¸�âƒ£ Attention mask patch (fix for off-by-one mask mismatch)
#         # ============================================================
#         import types
#         def patch_attention(layer):
#             if not hasattr(layer, "self_attn"):
#                 return
#             orig_forward = layer.self_attn.forward
#             def safe_forward(self, *args, **kwargs):
#                 attn_mask = kwargs.get("attention_mask", None)
#                 if attn_mask is not None and attn_mask.dim() == 4:
#                     bsz, _, q_len, kv_len = attn_mask.shape
#                     if kv_len < q_len:
#                         pad = torch.ones(
#                             (bsz, 1, q_len, q_len - kv_len),
#                             dtype=attn_mask.dtype,
#                             device=attn_mask.device,
#                         )
#                         attn_mask = torch.cat([attn_mask, pad], dim=-1)
#                         kwargs["attention_mask"] = attn_mask
#                     elif kv_len > q_len:
#                         kwargs["attention_mask"] = attn_mask[..., :q_len]
#                 return orig_forward(*args, **kwargs)
#             layer.self_attn.forward = types.MethodType(safe_forward, layer.self_attn)

#         for layer in getattr(model.model, "layers", []):
#             patch_attention(layer)
#         print("âœ… Patched DeepSeek attention mask mismatch")

#         # ============================================================
#         # Finalize model
#         # ============================================================
#         model.eval()
#         print("âœ… DeepSeek model ready on CPU")
#         return model, tokenizer

#     except Exception as e:
#         print(f"CRITICAL ERROR: Failed to load model: {e}")
#         print(traceback.format_exc())
#         return None, None

# def parse_llm_code(text):
#     m = re.search(r"```python(.*?)```", text, re.DOTALL)
#     if m: return m.group(1).strip()
#     m = re.search(r"def solve\(grid\).*", text, re.DOTALL)
#     return m.group(0).strip() if m else None

# LLM_META_PROMPT = """
# You are an ARC solver. Infer the transformation rule and write Python code.

# def solve(grid: list[list[int]]) -> list[list[int]]:
#     # your logic
# """

# def solve_with_llm(model, tokenizer, task, grid):
#     if not model or not tokenizer: return None
#     try:
#         train_text = "\n".join([
#             f"Example {i+1}\nInput:\n{p['input']}\nOutput:\n{p['output']}"
#             for i, p in enumerate(task["train"])
#         ])
#         prompt = f"{LLM_META_PROMPT}\n\nTrain:\n{train_text}\n\nTest:\n{grid}"
#         inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             temperature=0.1,
#             do_sample=False,
#             num_beams=1
#         )
#         text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         code = parse_llm_code(text)
#         if not code:
#             print("    -> LLM FAIL: no code")
#             return None

#         ns = {"np": np}
#         exec(code, ns)
#         func = ns.get("solve")
#         if not callable(func):
#             return None
#         result = func(grid)
#         print("    -> LLM executed successfully.")
#         return result
#     except Exception as e:
#         print(f"    -> LLM CRASH: {e}")
#         return None

# # --- Main -------------------------------------------------------------------
# def main():
#     print("="*40)
#     print("ARC 2025 - CPU Solver")
#     print("="*40)

#     model, tokenizer = load_model()
#     if not model:
#         print("âš ï¸� Running in heuristic-only mode.")

#     with open(TEST_CHALLENGES_FILE) as f:
#         test_data = json.load(f)
#     with open(SAMPLE_SUBMISSION_FILE) as f:
#         sample_sub = json.load(f)

#     submission = {}
#     for idx, tid in enumerate(sample_sub.keys(), 1):
#         print(f"\n--- Task {idx}/{len(sample_sub)}: {tid} ---")
#         task = test_data.get(tid)
#         if not task:
#             submission[tid] = sample_sub[tid]
#             continue

#         preds = []
#         predictor = None
#         for fn in HEURISTIC_SOLVERS:
#             p = check_heuristic(fn, task)
#             if p:
#                 predictor = p
#                 print(f"  -> Using heuristic: {fn.__name__}")
#                 break

#         for test_case in task["test"]:
#             inp = test_case["input"]
#             fallback = copy.deepcopy(inp)
#             pred1 = pred2 = None

#             if predictor:
#                 try:
#                     pred1 = predictor(inp)
#                 except Exception:
#                     pred1 = None

#             llm_pred = solve_with_llm(model, tokenizer, task, inp)
#             if pred1 is None:
#                 pred1 = llm_pred
#             else:
#                 pred2 = llm_pred

#             final1 = validate_grid(pred1, fallback)
#             final2 = validate_grid(pred2, fallback) if pred2 else fallback

#             preds.append({"attempt_1": final1, "attempt_2": final2})

#         submission[tid] = preds

#     with open(SUBMISSION_FILE, "w") as f:
#         json.dump(submission, f)
#     print("âœ… submission.json created on CPU.")

# if __name__ == "__main__":
#     main()



# ################################################################################
# # ARC 2025 - CPU-ONLY VERSION (CodeLlama 7B)
# ################################################################################

# import json
# import os
# import numpy as np
# import copy
# from pathlib import Path
# import time
# import re
# import traceback

# # --- Essential Imports -------------------------------------------------------
# try:
#     import torch
#     import transformers
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     print(f"Transformers version: {transformers.__version__}")
#     print(f"Torch version: {torch.__version__}")
# except ImportError:
#     print("Transformers or Torch not found. LLM will not work.")
#     AutoModelForCausalLM = None
#     torch = None

# try:
#     from scipy.ndimage import label
#     print("Scipy loaded successfully.")
# except ImportError:
#     print("Scipy not found. Object detection heuristics disabled.")
#     label = None

# # --- Configuration -----------------------------------------------------------
# DATA_DIR = Path("/kaggle/input/arc-prize-2025")
# MODEL_DIR = Path("/kaggle/input/codellama-7b/other/default/1")  # âœ… Use CodeLlama path
# TEST_CHALLENGES_FILE = DATA_DIR / "arc-agi_test_challenges.json"
# SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.json"
# SUBMISSION_FILE = "submission.json"

# device = "cpu"

# # --- Grid Validation ---------------------------------------------------------
# def validate_grid(grid, default_grid):
#     try:
#         if not isinstance(grid, list):
#             return default_grid
#         if len(grid) == 0 or len(grid) > 30:
#             return default_grid
#         width = len(grid[0])
#         if width == 0 or width > 30:
#             return default_grid

#         out = []
#         for r_idx, row in enumerate(grid):
#             if not isinstance(row, list) or len(row) != width:
#                 return default_grid
#             new_row = []
#             for c_idx, cell in enumerate(row):
#                 try:
#                     val = int(cell)
#                     if not (0 <= val <= 9):
#                         val = default_grid[r_idx][c_idx] % 10
#                     new_row.append(val)
#                 except Exception:
#                     return default_grid
#             out.append(new_row)
#         return out
#     except Exception:
#         return default_grid

# # --- Heuristic Solvers -------------------------------------------------------
# def get_objects(grid_np):
#     if label is None:
#         return [], 0
#     labeled, n = label(grid_np != 0)
#     objs = []
#     for i in range(1, n + 1):
#         mask = (labeled == i)
#         coords = np.argwhere(mask)
#         colors, counts = np.unique(grid_np[mask], return_counts=True)
#         objs.append({
#             "mask": mask,
#             "coords": coords,
#             "color": int(colors[np.argmax(counts)]),
#             "size": len(coords)
#         })
#     objs.sort(key=lambda o: o["size"], reverse=True)
#     return objs, n

# def _solve_identity(x): return x
# def _solve_flip_h(x): return np.fliplr(x)
# def _solve_flip_v(x): return np.flipud(x)
# def _solve_rotate_90(x): return np.rot90(x, k=1)
# def _solve_rotate_180(x): return np.rot90(x, k=2)
# def _solve_rotate_270(x): return np.rot90(x, k=3)
# def _solve_keep_largest_object(x):
#     objs, _ = get_objects(x)
#     if not objs: return x
#     g = np.zeros_like(x)
#     g[objs[0]["mask"]] = objs[0]["color"]
#     return g
# def _solve_keep_smallest_object(x):
#     objs, _ = get_objects(x)
#     if not objs: return x
#     g = np.zeros_like(x)
#     g[objs[-1]["mask"]] = objs[-1]["color"]
#     return g

# HEURISTIC_SOLVERS = [
#     _solve_identity, _solve_flip_h, _solve_flip_v,
#     _solve_rotate_90, _solve_rotate_180, _solve_rotate_270,
#     _solve_keep_largest_object, _solve_keep_smallest_object,
# ]

# def check_heuristic(solver_func, task):
#     try:
#         for p in task["train"]:
#             inp = np.array(p["input"], dtype=int)
#             out = solver_func(inp).tolist()
#             if out != p["output"]:
#                 return None
#         return lambda x: solver_func(np.array(x, dtype=int)).tolist()
#     except Exception:
#         return None

# # --- LLM Solver (CodeLlama on CPU) -------------------------------------------
# def load_model():
#     """Loads the CodeLlama model and tokenizer on CPU."""
#     if AutoModelForCausalLM is None:
#         print("Cannot load model, transformers library not found.")
#         return None, None

#     if not MODEL_DIR.exists():
#         print(f"CRITICAL ERROR: Model directory not found: {MODEL_DIR}")
#         return None, None

#     print(f"Loading CodeLlama model from: {MODEL_DIR}")
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_DIR,
#             torch_dtype=torch.float32,   # CPU friendly
#             device_map={"": "cpu"},
#             low_cpu_mem_usage=True,
#             trust_remote_code=True
#         ).to("cpu")

#         # âœ… Patch DynamicCache for Transformers â‰¥4.52
#         from transformers.cache_utils import DynamicCache
#         if not hasattr(DynamicCache, "get_max_length"):
#             def get_max_length(self):
#                 return self.get_seq_length() if hasattr(self, "get_seq_length") else 0
#             DynamicCache.get_max_length = get_max_length
#             print("âœ… Patched DynamicCache.get_max_length for Transformers â‰¥4.52")

#         model.eval()
#         print("âœ… CodeLlama model loaded and ready on CPU.")
#         return model, tokenizer

#     except Exception as e:
#         print(f"CRITICAL ERROR: Failed to load model: {e}")
#         print(traceback.format_exc())
#         return None, None

# # --- LLM Logic ---------------------------------------------------------------
# def parse_llm_code(text):
#     m = re.search(r"```python(.*?)```", text, re.DOTALL)
#     if m: return m.group(1).strip()
#     m = re.search(r"def solve\(grid\).*", text, re.DOTALL)
#     return m.group(0).strip() if m else None

# LLM_META_PROMPT = """<s>[INST] <<SYS>>
# You are an expert Python developer solving Abstraction and Reasoning Corpus (ARC) puzzles.
# Infer the transformation rule from the training examples (input â†’ output)
# and write a single, self-contained Python function that performs this transformation.
# Your output MUST include a ```python code block``` containing only the code.
# <</SYS>>

# Write a function following this signature:
# ```python
# import numpy as np
# def solve(grid: list[list[int]]) -> list[list[int]]:
#     # your logic here

# [/INST]
# """

# def solve_with_llm(model, tokenizer, task, grid):
#     if not model or not tokenizer:
#         return None
#     try:
#         train_text = "\n".join([
#             f"Example {i+1}\nInput:\n{p['input']}\nOutput:\n{p['output']}"
#             for i, p in enumerate(task["train"])
#         ])
#         prompt = f"""{LLM_META_PROMPT}
#                     ### TRAIN EXAMPLES:
#                     {train_text}
                    
#                     ### TEST INPUT:
#                     {grid}
                    
#                     ### RESPONSE:"""

#         inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=1024,
#             do_sample=False,
#             temperature=0.1,
#             top_p=0.9,
#             num_beams=1,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id
#         )

#         text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         code = parse_llm_code(text)
#         if not code:
#             print("    -> LLM FAIL: No code block found.")
#             return None

#         ns = {"np": np}
#         exec(code, ns)
#         func = ns.get("solve")
#         if not callable(func):
#             print("    -> LLM FAIL: No valid 'solve' function.")
#             return None

#         result = func(grid)
#         print("    -> LLM executed successfully.")
#         return result

#     except Exception as e:
#         print(f"    -> LLM CRASH: {e}")
#         return None

# # --- Main Execution ----------------------------------------------------------
# def main():
#     print("="*40)
#     print("ARC 2025 - CPU Solver (CodeLlama 7B)")
#     print("="*40)

#     model, tokenizer = load_model()
#     if not model:
#         print("âš ï¸� Running in heuristic-only mode.")

#     with open(TEST_CHALLENGES_FILE) as f:
#         test_data = json.load(f)
#     with open(SAMPLE_SUBMISSION_FILE) as f:
#         sample_sub = json.load(f)

#     submission = {}
#     for idx, tid in enumerate(sample_sub.keys(), 1):
#         print(f"\n--- Task {idx}/{len(sample_sub)}: {tid} ---")
#         task = test_data.get(tid)
#         if not task:
#             submission[tid] = sample_sub[tid]
#             continue

#         preds = []
#         predictor = None
#         for fn in HEURISTIC_SOLVERS:
#             p = check_heuristic(fn, task)
#             if p:
#                 predictor = p
#                 print(f"  -> Using heuristic: {fn.__name__}")
#                 break

#         for test_case in task["test"]:
#             inp = test_case["input"]
#             fallback = copy.deepcopy(inp)
#             pred1 = pred2 = None

#             if predictor:
#                 try:
#                     pred1 = predictor(inp)
#                 except Exception:
#                     pred1 = None

#             llm_pred = solve_with_llm(model, tokenizer, task, inp)
#             if pred1 is None:
#                 pred1 = llm_pred
#             else:
#                 pred2 = llm_pred

#             final1 = validate_grid(pred1, fallback)
#             final2 = validate_grid(pred2, fallback) if pred2 else fallback

#             preds.append({"attempt_1": final1, "attempt_2": final2})

#         submission[tid] = preds

#     with open(SUBMISSION_FILE, "w") as f:
#         json.dump(submission, f)
#     print("âœ… submission.json created on CPU with CodeLlama 7B.")

# if __name__ == "__main__":
#     main()



################################################################################
# ARC 2025 - GPU VERSION (CodeLlama 7B)
################################################################################

import json
import os
import numpy as np
import copy
from pathlib import Path
import time
import re
import traceback

# --- Essential Imports -------------------------------------------------------
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Transformers version: {transformers.__version__}")
    print(f"Torch version: {torch.__version__}")
except ImportError:
    print("Transformers or Torch not found. LLM will not work.")
    AutoModelForCausalLM = None
    torch = None

try:
    from scipy.ndimage import label
    print("Scipy loaded successfully.")
except ImportError:
    print("Scipy not found. Object detection heuristics disabled.")
    label = None

# --- Configuration -----------------------------------------------------------
DATA_DIR = Path("/kaggle/input/arc-prize-2025")
MODEL_DIR = Path("/kaggle/input/codellama-7b/other/default/1")
TEST_CHALLENGES_FILE = DATA_DIR / "arc-agi_test_challenges.json"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.json"
SUBMISSION_FILE = "submission.json"

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"âœ… Using device: {device}")

# --- Grid Validation ---------------------------------------------------------
def validate_grid(grid, default_grid):
    try:
        if not isinstance(grid, list) or len(grid) == 0 or len(grid) > 30:
            return default_grid
        width = len(grid[0])
        if width == 0 or width > 30:
            return default_grid
        out = []
        for r_idx, row in enumerate(grid):
            if not isinstance(row, list) or len(row) != width:
                return default_grid
            new_row = []
            for c_idx, cell in enumerate(row):
                try:
                    val = int(cell)
                    if not (0 <= val <= 9):
                        val = default_grid[r_idx][c_idx] % 10
                    new_row.append(val)
                except Exception:
                    return default_grid
            out.append(new_row)
        return out
    except Exception:
        return default_grid

# --- Heuristic Solvers -------------------------------------------------------
def get_objects(grid_np):
    if label is None:
        return [], 0
    labeled, n = label(grid_np != 0)
    objs = []
    for i in range(1, n + 1):
        mask = (labeled == i)
        coords = np.argwhere(mask)
        colors, counts = np.unique(grid_np[mask], return_counts=True)
        objs.append({
            "mask": mask,
            "coords": coords,
            "color": int(colors[np.argmax(counts)]),
            "size": len(coords)
        })
    objs.sort(key=lambda o: o["size"], reverse=True)
    return objs, n

def _solve_identity(x): return x
def _solve_flip_h(x): return np.fliplr(x)
def _solve_flip_v(x): return np.flipud(x)
def _solve_rotate_90(x): return np.rot90(x, k=1)
def _solve_rotate_180(x): return np.rot90(x, k=2)
def _solve_rotate_270(x): return np.rot90(x, k=3)
def _solve_keep_largest_object(x):
    objs, _ = get_objects(x)
    if not objs: return x
    g = np.zeros_like(x)
    g[objs[0]["mask"]] = objs[0]["color"]
    return g
def _solve_keep_smallest_object(x):
    objs, _ = get_objects(x)
    if not objs: return x
    g = np.zeros_like(x)
    g[objs[-1]["mask"]] = objs[-1]["color"]
    return g

HEURISTIC_SOLVERS = [
    _solve_identity, _solve_flip_h, _solve_flip_v,
    _solve_rotate_90, _solve_rotate_180, _solve_rotate_270,
    _solve_keep_largest_object, _solve_keep_smallest_object,
]

def check_heuristic(solver_func, task):
    try:
        for p in task["train"]:
            inp = np.array(p["input"], dtype=int)
            out = solver_func(inp).tolist()
            if out != p["output"]:
                return None
        return lambda x: solver_func(np.array(x, dtype=int)).tolist()
    except Exception:
        return None

# --- LLM Solver (GPU CodeLlama) ----------------------------------------------
def load_model():
    """Loads CodeLlama model and tokenizer on GPU."""
    if AutoModelForCausalLM is None:
        print("Cannot load model, transformers library not found.")
        return None, None

    if not MODEL_DIR.exists():
        print(f"CRITICAL ERROR: Model directory not found: {MODEL_DIR}")
        return None, None

    print(f"ğŸš€ Loading CodeLlama model from: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # âœ… Patch DynamicCache for Transformers â‰¥4.52
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "get_max_length"):
            def get_max_length(self):
                return self.get_seq_length() if hasattr(self, "get_seq_length") else 0
            DynamicCache.get_max_length = get_max_length
            print("âœ… Patched DynamicCache.get_max_length for Transformers â‰¥4.52")

        model.eval()
        print("âœ… CodeLlama model loaded and ready on GPU.")
        return model, tokenizer

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model: {e}")
        print(traceback.format_exc())
        return None, None

# --- LLM Logic ---------------------------------------------------------------
def parse_llm_code(text: str):
    """
    Extracts, cleans, and validates Python code from LLM output.
    Handles Unicode, junk text, and missing indentation gracefully.
    """

    if not text or not isinstance(text, str):
        return None
    # print(text)
    # --- Step 1: Remove markdown and junk markers
    junk_tokens = [
        "```python", "```", "code block", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
        "</s>", "<s>", "python", "Train examples:", "Test input:", "Output:"
    ]
    for tok in junk_tokens:
        text = text.replace(tok, "")
    text = re.sub(r"<.*?>", "", text)

    # --- Step 2: Replace bad unicode characters
    text = (text
        .replace("â†’", "->")
        .replace("â†�", "<-")
        .replace("â€¢", "*")
        .replace("â€“", "-")
    )

    # --- Step 3: Find the def solve() block
    match = re.search(r"(def\s+solve\s*\(.*?\):[\s\S]*)", text)
    code = match.group(1) if match else text

    # --- Step 4: Remove natural language junk
    clean_lines = []
    for line in code.splitlines():
        l = line.strip()
        if not l:
            continue
        if any(k in l.lower() for k in ["example", "train", "test", "input", "output", "infer", "rule"]):
            continue
        if re.match(r"^[#>*\-].*", l):
            continue
        if re.search(r"[a-zA-Z]{4,}\s+[a-zA-Z]{4,}", l) and not l.strip().startswith("#"):
            continue
        clean_lines.append(line)

    code = "\n".join(clean_lines).strip()

    # --- Step 5: Ensure function body exists
    if re.match(r"^def\s+solve\s*\(.*?\):\s*$", code):
        # If function is empty, add a safe body
        code += "\n    return grid"

    elif "def solve(" not in code:
        # wrap orphaned code in function
        wrapped = "\n".join("    " + l for l in code.splitlines() if l.strip())
        code = f"def solve(grid):\n{wrapped or '    return grid'}"

    # --- Step 6: Add numpy import at top
    if not code.strip().startswith("import numpy"):
        code = "import numpy as np\n" + code

    # --- Step 7: Sanitize and verify
    code = re.sub(r"\n{3,}", "\n\n", code).strip()
    try:
        compile(code, "<string>", "exec")
    except Exception as e:
        print(f"âš ï¸�  parse_llm_code(): cleaned code still invalid: {e}")
        return "import numpy as np\ndef solve(grid):\n    return grid"

    return code



LLM_META_PROMPT = """<s>[INST] <<SYS>>
You are an expert Python developer solving Abstraction and Reasoning Corpus (ARC) puzzles.
Infer the transformation rule from the training examples (input â†’ output)
and write a single, self-contained Python function that performs this transformation.
Your output MUST include a ```python code block``` containing only the code.
Remember: Always include valid, executable Python code inside a ```python``` block
there should onky be executable python code nothing else.
<</SYS>>

Write a function following this signature:
```python
import numpy as np
def solve(grid: list[list[int]]) -> list[list[int]]:
    # your logic here

[/INST]"""

def solve_with_llm(model, tokenizer, task, grid):
    if not model or not tokenizer:
        return None
    try:
        train_text = "\n".join([
            f"Example {i+1}\nInput:\n{p['input']}\nOutput:\n{p['output']}"
            for i, p in enumerate(task["train"])
        ])
        prompt = f"""{LLM_META_PROMPT}
Train examples:
{train_text}

Test input:
{grid}
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
        # print(text)
        code = parse_llm_code(text)
        print(code)
        if not code:
            print("    -> âš ï¸� LLM FAIL: No code block found.")
            return None

        ns = {"np": np}
        exec(code, ns)
        func = ns.get("solve")
        if not callable(func):
            print("    -> âš ï¸� LLM FAIL: No valid 'solve' function.")
            return None

        result = func(grid)
        print("    -> âœ… LLM executed successfully.")
        return result

    except Exception as e:
        print(f"    -> â�Œ LLM CRASH: {e}")
        return None

# --- Main Execution ----------------------------------------------------------
def main():
    print("="*40)
    print("ARC 2025 - GPU Solver (CodeLlama 7B)")
    print("="*40)

    model, tokenizer = load_model()
    if not model:
        print("âš ï¸� Running in heuristic-only mode.")

    with open(TEST_CHALLENGES_FILE) as f:
        test_data = json.load(f)
    with open(SAMPLE_SUBMISSION_FILE) as f:
        sample_sub = json.load(f)

    submission = {}
    for idx, tid in enumerate(sample_sub.keys(), 1):
        print(f"\n--- Task {idx}/{len(sample_sub)}: {tid} ---")
        task = test_data.get(tid)
        if not task:
            submission[tid] = sample_sub[tid]
            continue

        preds = []
        predictor = None
        for fn in HEURISTIC_SOLVERS:
            p = check_heuristic(fn, task)
            if p:
                predictor = p
                print(f"  -> âœ… Using heuristic: {fn.__name__}")
                break

        for test_case in task["test"]:
            inp = test_case["input"]
            fallback = copy.deepcopy(inp)
            pred1 = pred2 = None

            if predictor:
                try:
                    pred1 = predictor(inp)
                except Exception:
                    pred1 = None

            llm_pred = solve_with_llm(model, tokenizer, task, inp)
            if pred1 is None:
                pred1 = llm_pred
            else:
                pred2 = llm_pred

            final1 = validate_grid(pred1, fallback)
            final2 = validate_grid(pred2, fallback) if pred2 else fallback

            preds.append({"attempt_1": final1, "attempt_2": final2})

        submission[tid] = preds

    with open(SUBMISSION_FILE, "w") as f:
        json.dump(submission, f)
    print("âœ… submission.json created using GPU + CodeLlama 7B.")

if __name__ == "__main__":
    main()





