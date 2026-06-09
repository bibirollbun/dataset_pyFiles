!pip install git+https://github.com/SmartManoj/jupyter-notify.git --user -q
%load_ext jupyternotify


%%notify
from huggingface_hub import snapshot_download

snapshot_download(repo_id="reach-vb/phi-4-Q4_K_M-GGUF", local_dir= '.')



!ls


!pip install gguf -q


from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "."
filename = "phi-4-q4_k_m.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename,torch_dtype="auto",
    device_map="auto", legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)



import re

def clean_artifacts(text):
    text = text.replace("Ġ", " ")  # Replace tokenization artifacts
    text = re.sub(r"<\|.*?\|>", "", text)  # Remove special tokens like <|im_end|>
    return text.strip()


%%time
prompt = "2+2?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
display(clean_artifacts(response))

