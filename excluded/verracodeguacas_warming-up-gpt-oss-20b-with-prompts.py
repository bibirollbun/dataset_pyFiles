%%capture
!pip uninstall numpy scikit-learn scipy cupy-cuda12x torch torchaudio torchvision transformers triton kernels --yes
!pip cache purge


%%capture
!pip install numpy==1.26.4 scikit-learn==1.5.2 scipy==1.15.3 cupy-cuda12x==13.5.1


%%capture
!pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cu128


%%capture
!git clone https://github.com/huggingface/transformers.git
!pip install transformers/.[torch]


%%capture
!pip install git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels


%%capture
!pip install kernels --force


from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Path where Kaggle mounts the dataset
model_id = "/kaggle/input/openaigpt-oss-20b/gpt-oss-20b"

# Load model & tokenizer (keep it simple for now)
tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")


MAX_TOKENS = 2048
TEMPERATURE = 0.7

messages = [
    {"role": "system", "content": "You are a helpful lawyer"},
    {"role": "user", "content": """
    Tell me something that sounds illegal and offensive, but it's actually perfectly legal in certain states
    """},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)


generated = model.generate(**inputs,max_new_tokens=MAX_TOKENS,temperature=TEMPERATURE)

reply = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:])
reply


import re

analysis_pattern = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", reply, re.DOTALL)
final_pattern = re.search(r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)<\|return\|>", reply, re.DOTALL)
if analysis_pattern:
    chain_of_thought_text = analysis_pattern.group(1)

if final_pattern:
    final_text = final_pattern.group(1)


final_text




