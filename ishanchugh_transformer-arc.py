!pip install --upgrade transformers --no-index --find-links="/kaggle/input/transformer-latest"


from transformers import pipeline
import torch

model_id = "/kaggle/input/gpt-oss-20b/transformers/default/1"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])




