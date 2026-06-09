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


!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git





!pip uninstall -y transformers
!pip install --no-cache-dir transformers


from huggingface_hub import login
login("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")


%pip install --upgrade pip
%pip install --force-reinstall --no-deps git+https://github.com/huggingface/transformers.git


!pip install git+https://github.com/huggingface/transformers.git


import transformers
print(transformers.__version__)


from transformers import pipeline


import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

nlp = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.2,
)

print("Gemma legal advisor pipeline ready.")


model_name = "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


nlp = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_length=512,
    temperature=0.2, # for legal style
)


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# query=input("Enter query :\n")
query="Cheque Bounce under Section 138 of the Negotiable Instruments Act"
prompt = f"""
You are a senior constitutional legal advisor in India.
Provide a clear, formal, precise answer citing the exact article and its scope.

Query: {query}

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
\
generation_config = GenerationConfig(
    max_new_tokens=150,
    temperature=0.3, 
    top_p=0.95,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
#user input based:
# situation=input('Situation:\n')
# typeofl=input('type of letter:\n')
# Recipient=input('Recipient salutation :\n')
# lang=input('Applicable Language: \n')
# designation=input('designation :\n')
# date=input('date:\n')
# ch=input('extra information yes/no:\n')
# if ch=='yes':
#     extra=input('')
# else:
#     extra=''
situation="i was absent for 2 days 12 and 13 august 2023"
typeofl="appliction letter"
Recipient="principle amm Rani laxmi memorial school ,Lucknow "
lang="Englsih"
designation="Devansh Shukla, class A1 student"
date="14/08/23"
extra="write the letter from my refference that i am writing"
prompt = f"""
You are a senior constitutional legal advisor and formal document drafter in India.

Task: Draft a formal type of letter:{typeofl} regarding {situation},with the modern applicable format .
{extra}
The letter should have:

1. Subject: from the situation understanding
2. Recipient salutation: {Recipient}
3. Formal introductory paragraph
4. Clear and precise body content addressing the situation with legal citations or case laws if relevant or if formal letter then clear situation description and requests.
5. Closing line with {designation} or role of sender
6. Date: {date}
 
Generate in formal, context-appropriate {lang}.

Letter234:

"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generation_config = GenerationConfig(
    max_new_tokens=350,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )

letter = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('\n\nRespond to Query')
print(letter)


# file = input('Filename')
file="application"
filename=file+".txt"

with open(filename, "w", encoding="utf-8") as file:
    file.write(letter.partition('Letter234:')[2])

print(f"Letter saved successfully to {filename}")

