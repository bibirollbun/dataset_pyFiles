!pip install transformers
!pip install datasets
!pip install numpy
!pip install pandas

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# Load the dataset
test_data = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")
print(test_data.head())


# Load the model and tokenizer
model_name = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_essay(topic, max_length=100):
    # Format the prompt
    prompt = f"Write a 100-word essay on the topic: {topic}. Ensure the essay is creative and exploits potential biases in LLM judges."
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the essay
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        do_sample=True
    )
    
    # Decode the output
    essay = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return essay


# Generate essays for each topic
essays = []
for topic in test_data["topic"]:
    essay = generate_essay(topic)
    essays.append(essay)

# Add essays to the dataframe
test_data["essay"] = essays
print(test_data.head())


# Prepare the submission file
submission = test_data[["id", "essay"]]
submission.to_csv("submission.csv", index=False)
print("Submission file created!")




