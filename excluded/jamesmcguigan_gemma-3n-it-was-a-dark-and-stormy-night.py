%%time
!pip install --quiet timm --upgrade 2> /dev/null
!pip install --quiet accelerate     2> /dev/null
!pip install --quiet git+https://github.com/huggingface/transformers.git 2> /dev/null


%%time
# Source: https://www.kaggle.com/models/google/gemma-3n

import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b")  # = 4 billion parameters
# GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b")  # = 2 billion parameters
processor  = AutoProcessor.from_pretrained(GEMMA_PATH)
model      = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

def ask_gemma(prompt):
    input_ids = processor(text=str(prompt), return_tensors="pt").to(model.device, dtype=model.dtype)
    outputs   = model.generate(**input_ids, max_new_tokens=512, disable_compile=True)
    text = processor.batch_decode(
        outputs,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )
    return text[0]  # text[0] injects <bos>prompt


%%time
prompt_history = ask_gemma("It was a dark and stormy night.")
print(prompt_history)
print()


%%time
prompt_history = ask_gemma(prompt_history)
print(prompt_history)
print()


prompt_history = ask_gemma(prompt_history)
print(prompt_history)
print()

