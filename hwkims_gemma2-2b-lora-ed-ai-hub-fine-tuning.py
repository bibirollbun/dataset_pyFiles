import os
# Keras Backend ì„¤ì •: JAXë¥¼ ê°€ì�¥ ë¨¼ì € ì„¤ì •í•´ì•¼ í•©ë‹ˆë‹¤.
os.environ["KERAS_BACKEND"] = "jax"
# ë©”ëª¨ë¦¬ fragmentation ë°©ì§€
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras
from jax import config

# Configure JAX to use float32 for better accuracy with Gemma
config.update("jax_default_matmul_precision", "float32")

# ëª¨ë�¸ ID ë°� LoRA ì„¤ì •
model_id = "gemma2_instruct_2b_en"
lora_rank = 4
lora_weights_path = "/kaggle/input/finetune/keras/default/1/korean_4_epoch1.lora.h5"
output_path = "/kaggle/working/my_fine_tuned_gemma2_full_rank4.keras" # Added the .keras extension

# ê¸°ë³¸ ëª¨ë�¸ ë¡œë“œ
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(model_id, dtype="bfloat16")

# LoRA í™œì„±í™”
gemma_lm.backbone.enable_lora(rank=lora_rank)

# LoRA ê°€ì¤‘ì¹˜ ë¡œë“œ
gemma_lm.backbone.load_lora_weights(lora_weights_path)
print(f"LoRA weights loaded from {lora_weights_path}")

# ì „ì²´ ëª¨ë�¸ ì €ì�¥
gemma_lm.save(output_path)
print(f"Fine-tuned Gemma 2 model with LoRA rank 4 saved to {output_path}")


