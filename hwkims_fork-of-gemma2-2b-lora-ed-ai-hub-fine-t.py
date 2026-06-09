import os
# Keras Backend ì„¤ì •: JAXë¥¼ ê°€ì�¥ ë¨¼ì € ì„¤ì •í•´ì•¼ í•©ë‹ˆë‹¤.
os.environ["KERAS_BACKEND"] = "jax" if os.environ.get("TPU_NAME") is None else "tensorflow"

# ë©”ëª¨ë¦¬ fragmentation ë°©ì§€
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras
from jax import config
import json
import safetensors
from safetensors.torch import save_file
import torch
import numpy as np
import sentencepiece as spm # SentencePiece import ì¶”ê°€


# JAX ì„¤ì • (TPU í™˜ê²½ì�´ ì•„ë‹Œ ê²½ìš°ì—�ë§Œ ì �ìš©)
if os.environ.get("TPU_NAME") is None:
  config.update("jax_default_matmul_precision", "float32")

# ëª¨ë�¸ ID ë°� LoRA ì„¤ì •
model_id = "gemma2_instruct_2b_en"
lora_rank = 4
lora_weights_path = "/kaggle/input/finetune/keras/default/1/korean_4_epoch1.lora.h5"
output_dir = "/kaggle/working/my_fine_tuned_gemma2_full_rank4"  # ìˆ˜ì •: ì €ì�¥í•  ë””ë ‰í† ë¦¬ ê²½ë¡œ (Kaggle /working)


os.makedirs(output_dir, exist_ok=True)  # ë””ë ‰í† ë¦¬ ìƒ�ì„±

# ê¸°ë³¸ ëª¨ë�¸ ë¡œë“œ
import keras_hub
gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(model_id, dtype="bfloat16")


# LoRA í™œì„±í™”
gemma_lm.backbone.enable_lora(rank=lora_rank)

# LoRA ê°€ì¤‘ì¹˜ ë¡œë“œ
gemma_lm.backbone.load_lora_weights(lora_weights_path)
print(f"LoRA weights loaded from {lora_weights_path}")

# --- ëª¨ë�¸ ê°€ì¤‘ì¹˜ ì¶”ì¶œ ë°� safetensorsë¡œ ì €ì�¥ ---
weights = gemma_lm.get_weights()
torch_dict = {}
for i, weight in enumerate(weights):
    # bfloat16ì�„ float32ë¡œ ë³€í™˜ í›„ PyTorch í…�ì„œë¡œ ë³€í™˜
    if weight.dtype == np.dtype('bfloat16'):
        weight = weight.astype(np.float32)
    torch_dict[f"param_{i}"] = torch.from_numpy(weight)

# ì–‘ì��í™” (ì„ íƒ� ì‚¬í•­, CPU í™˜ê²½ì—�ì„œ)
if os.environ.get("TPU_NAME") is None:
    torch_dict = {key: tensor.half() for key, tensor in torch_dict.items()}
    print("FP16 ì–‘ì��í™” ì �ìš©")
    
save_file(torch_dict, os.path.join(output_dir, "model.safetensors"))
print("ëª¨ë�¸ ê°€ì¤‘ì¹˜ ì €ì�¥ ì™„ë£Œ: model.safetensors")

# --- ëª¨ë�¸ ì•„í‚¤í…�ì²˜ ì¶”ì¶œ ë°� config.jsonìœ¼ë¡œ ì €ì�¥ ---
model_config = gemma_lm.get_config()
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(model_config, f, indent=4)
print("ëª¨ë�¸ ì•„í‚¤í…�ì²˜ ì €ì�¥ ì™„ë£Œ: config.json")


# --- í† í�¬ë‚˜ì�´ì € ì •ë³´ ì €ì�¥ (ì�…ë ¥ ëª¨ë�¸ì—�ì„œ í† í�¬ë‚˜ì�´ì € ë¡œë“œ) ---
if hasattr(gemma_lm, 'preprocessor') and hasattr(gemma_lm.preprocessor, 'tokenizer') :
    tokenizer = gemma_lm.preprocessor.tokenizer
    if hasattr(tokenizer, 'spm_tokenizer') :
        tokenizer_config = tokenizer.get_config()
        # vocabulary.spm íŒŒì�¼ ìƒ�ì„±
        spm_model_path = os.path.join(output_dir, "vocabulary.spm")
        tokenizer.spm_tokenizer.save_model(spm_model_path)
        print("í† í�¬ë‚˜ì�´ì € vocabulary.spm ì €ì�¥ ì™„ë£Œ")
        # tokenizer.json íŒŒì�¼ ì €ì�¥
        with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding='utf-8') as f:
            json.dump(tokenizer_config, f, indent=4, ensure_ascii=False)
        print("í† í�¬ë‚˜ì�´ì € config ì €ì�¥ ì™„ë£Œ: tokenizer.json")
    else :
      print("í•´ë‹¹ ëª¨ë�¸ì�€ sentencepieceë¥¼ ì‚¬ìš©í•˜ì§€ ì•ŠìŠµë‹ˆë‹¤.")

else :
    print("ëª¨ë�¸ì—� í† í�¬ë‚˜ì�´ì € ì •ë³´ê°€ ì—†ìŠµë‹ˆë‹¤.")


print(f"Fine-tuned Gemma 2 model with LoRA rank 4 saved to {output_dir}")

