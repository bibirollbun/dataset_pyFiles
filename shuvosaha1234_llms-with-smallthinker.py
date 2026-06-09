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


import os
import torch
import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import Markdown

# Define the model path
model_path = "/kaggle/input/phi-3/pytorch/phi-3.5-mini-instruct/2"

# Inspect the model directory structure
print("Inspecting model directory...")
for root, dirs, files in os.walk(model_path):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}\n")

# Load the model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",  # Automatically use the best dtype for the hardware
    device_map="auto"    # Automatically assign model to available devices
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the essay prompt template
prompt = """
Write a 100-word essay on the topic: {topic}. The essay should:
1. Present a thought-provoking argument that is open to multiple interpretations.
2. Include controversial or polarizing ideas to challenge the judges' perspectives.
3. Use a mix of formal and informal language to exploit style biases.
4. Avoid repetition and ensure the essay is written in clear, grammatically correct English.
5. Appeal to the judges' self-awareness by referencing their role as LLMs in shaping AI ethics.
## Topic: {topic}
## Essay\n
"""

# Function to generate essays
def get_essay(topic, should_display=False):
    question = prompt.replace("{topic}", topic)
    if should_display:
        display(Markdown("## Question"))
        display(Markdown(question))
    input_ids = tokenizer.encode(question, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=192,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text.split("## Essay\n")[-1]
    if should_display:
        display(Markdown(answer))
    return answer

# Read the test CSV file
print("Reading test CSV...")
test_path = "/kaggle/input/llms-you-cant-please-them-all/test.csv"
test = pd.read_csv(test_path)
print(test.head())

# Generate essays
print("Generating essays...")
essays = []
for i in tqdm(range(len(test))):
    essay = get_essay(test.iloc[i]["topic"], should_display=i <= 3)
    essays.append(essay)

# Save the results to a new CSV file
test["essay"] = essays
output_path = "submission.csv"
test[["id", "essay"]].to_csv(output_path, index=False)
print("Essays saved to:", output_path)

# Display the first few results
print(test[["id", "essay"]].head())


