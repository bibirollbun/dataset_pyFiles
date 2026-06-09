!ls /kaggle/input/qwen-3/transformers/0.6b/1


!du -sh /kaggle/input/prepare-arc-prize-2025-offline-packages


!pip install --no-index --find-links=/kaggle/input/prepare-arc-prize-2025-offline-packages -U torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 trl==0.17.0 bitsandbytes==0.45.5 vllm==0.8.5


!pip show transformers trl bitsandbytes torch


import time
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/kaggle/input/qwen-3/transformers/0.6b/1"
# MODEL_PATH = "/kaggle/input/qwen-3/transformers/4b/1"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained('.')
# , quantization_config=nf4_config  , quantization_config=nf4_config device_map={"":0},
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"":0},
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)
model.save_pretrained('.')

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

torch.cuda.synchronize()
t0 = time.time()
# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
torch.cuda.synchronize()
t1 = time.time()

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
print(f"[~] time: {round(t1 - t0, 2)} s, tokens: {len(output_ids)}, avg: {round(1000*(t1 - t0) / len(output_ids), 2)} ms")
# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)


del model, model_inputs, generated_ids
import gc
gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
!du -sh .
!du -sh /kaggle/input/qwen-3/transformers/0.6b/1

