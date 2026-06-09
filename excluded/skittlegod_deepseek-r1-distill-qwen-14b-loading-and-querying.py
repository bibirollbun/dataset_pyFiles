!pip install --no-index --find-links=/kaggle/input/bitsandbytes-package-offline/kaggle/working/packs bitsandbytes


import bitsandbytes as bnb
import numpy as np 
import pandas as pd 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import re


sample_df = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv')
sample_df


def remove_non_english(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)


def generate_essay(prompt: str, max_length: int = 250) -> str:
    

    model.eval()

    
    # Ensure no gradients are calculated during tokenization or generation
    with torch.no_grad():
        sys_prompt = "The goal is to write an essay with no more than 100 words to maximize disagreement between three individual LLM-judges, while using the English language, and without repeating yourself. The topic is: "
        prompt = sys_prompt + prompt + " ###"
        
        # Tokenize input with attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generate text with attention mask
        output_ids = model.generate(
            input_ids, 
            attention_mask=attention_mask,  # Pass attention mask
            max_length=max_length, 
            temperature=0.1, 
            top_p=0.9, 
            do_sample=True,
            num_return_sequences=1,  # Return only one sequence for faster results
            no_repeat_ngram_size=2  # Prevent repeated n-grams for diversity
        )

        # Decode and return generated text
        essay = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        essay = essay.split("###",1)[1]
    return remove_non_english(essay)


quant_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


model_name = "/kaggle/input/deepseek-r1-distill-qwen-14b/transformers/default/1/R1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")


# model = torch.compile(model)


def create_submission(test_df):
    submissions = [{'id': row['id'], 'essay': generate_essay(row['topic'])} for _, row in test_df.iterrows()]
    return pd.DataFrame(submissions)


def save_submission(submission_df, output_file):
    submission_df.to_csv(output_file, index=False)


test_data = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')

submission_df = create_submission(test_data)

save_submission(submission_df, "submission.csv")


print(submission_df['essay'][0])




