# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Path to locally stored Qwen model
model_path = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"

# Load model as a standard transformer model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True  # Some models require this to load properly
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


test = pd.read_csv(r'/kaggle/input/llms-you-cant-please-them-all/test.csv')


from sentence_transformers import SentenceTransformer

# Path to the locally stored model
embedding_model_path = "/kaggle/input/m/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

# Load the model correctly
embedding_model = SentenceTransformer(embedding_model_path, local_files_only=True)


topics = test['topic'].to_list()


topics


def generate_essay(topics, llm_pipeline):
    responses = []
    for topic in topics:
        # Create a prompt
        prompt = f"""Write a concise essay of exactly 100 words on the topic of your choice.
        The essay should include a clear introduction, a body with supporting details, and a conclusion.
        Ensure that the language is formal and appropriate for an academic audience.
        Use varied sentence structures to enhance the flow and readability of the text.
        Emphasize clarity and coherence, making sure each sentence logically connects to the next.
        Avoid unnecessary jargon or complex vocabulary that may confuse the reader.
        Proofread for grammatical accuracy and proper punctuation.
        Finally, submit the essay in a well-organized format.\n\n Topic: {topic}
        \n\nAnswer:"""
        
        # Generate a response using the LLM
        response = llm_pipeline(prompt, max_length=1000, 
                                num_return_sequences=1, 
                                truncation=True,
                                temperature=0.5
                                )[0]['generated_text']
        
        print(f"Topic: {topic}\nResponse: {response}\n{'-'*50}")
        responses.append(response)

    return responses


essays = generate_essay(topics, llm_pipeline)


print(essays)


import re

def extract_after_think(essays):
    sub = []
    for text in essays:
        parts = text.split("Answer:", 1)
        essay = parts[1].strip() if len(parts) > 1 else ""
        essay = re.sub(r'[\n\r*]', '', essay)
        sub.append(essay)
    return sub



sub = extract_after_think(essays)


sub


submission = pd.read_csv(r'/kaggle/input/llms-you-cant-please-them-all/sample_submission.csv')


submission


submission['essay'] = sub


submission


submission.to_csv("submission.csv", index=False)




