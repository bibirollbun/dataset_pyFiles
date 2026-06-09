# Inference using transformers library
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# Model configuration
BASE_MODEL = "/kaggle/input/qwen3-4b"
LORA_PATH = "/kaggle/input/qwen3-4b-ft-jigsaw/trained_model"

# Load test data
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# Define system prompt
SYS_PROMPT = """
You are given a comment on reddit. Your task is to classify if it violates the given rule. Only respond Yes/No.
"""

# Function to get predictions
def get_prediction(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get logits for the first generated token
    logits = outputs.scores[0][0]
    
    # Get token IDs for "Yes" and "No"
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    # Extract logits for Yes/No
    yes_logit = logits[yes_id].item()
    no_logit = logits[no_id].item()
    
    # Apply softmax
    probs = torch.nn.functional.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
    
    return probs[0].item()  # Return probability of "Yes"

# Prepare predictions
predictions = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = f"""
r/{row.subreddit}
Rule: {row.rule}
1) {row.positive_example_1}
Violation: Yes
2) {row.negative_example_1}
Violation: No
3) {row.negative_example_2}
Violation: No
4) {row.positive_example_2}
Violation: Yes
5) {row.body}
"""
    
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    ) + "Answer:"
    
    pred = get_prediction(prompt, model, tokenizer)
    predictions.append(pred)

# Create submission
df['rule_violation'] = predictions
df[['row_id', 'rule_violation']].to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
df[['row_id', 'rule_violation']]




