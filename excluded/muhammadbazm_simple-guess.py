# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_test = pd.read_csv("/kaggle/input/llms-you-cant-please-them-all/test.csv")

print(df_test.shape)
# print(df_test)
print(df_test.columns)


from time import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from IPython.display import display, Markdown


model_address = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"
tokenizer = AutoTokenizer.from_pretrained(model_address)
model = AutoModelForCausalLM.from_pretrained(model_address)


pipe = pipeline("text-generation", 
                model=model, 
                tokenizer=tokenizer
               )


perspectives = [
    "Ethical Perspective",
    "Practical / Technical Perspective",
    "Risk & Safety Perspective",
    "Economic Perspective",
    "Regulatory / Policy Perspective",
    "Societal Perspective",
    "Philosophical / Conceptual Perspective",
    "Psychological Perspective",
    "Humanistic / Empathy Perspective",
    "Environmental Perspective",
    "Futurist / Forward-Looking Perspective",
    "Equity & Social Justice Perspective",
    "Corporate / Business Perspective",
    "Legal / Jurisprudence Perspective",
    "Historical Perspective",
    "Metaphysical / Spiritual Perspective"
]

tones = [
    "Formal",
    "Informal",
    "Conversational",
    "Authoritative",
    "Objective / Neutral",
    "Subjective / Personal",
    "Emotional / Emotive",
    "Humorous",
    "Satirical",
    "Sarcastic",
    "Inspirational / Uplifting",
    "Confrontational",
    "Diplomatic",
    "Provocative"
]

styles = [
    "Academic",
    "Technical",
    "Persuasive",
    "Reflective",
    "Analytical",
    "Instructional / Didactic",
    "Descriptive",
    "Narrative",
    "Expository",
    "Poetic",
    "Journalistic"
]

def get_item(items_list):
    index = random.randint(0, len(items_list)-1)
    return items_list[index]


topic = "wealth"

prompt = f"""
Generate a piece of text about {topic} message with the following considerations: 
1- Make sure it doesn't exceed 100 words.
2- Add a few conflicting view points.
3- Ensure the text is in clear and fluent English.
4- Generate the text using a {get_item(perspectives)} perspective.
5- Please generate the text with a / an {get_item(tones)} tone.
6- Please generate the text with a / an {get_item(styles)} style.
*
"""

# Generate text using the pipeline
generated_texts = pipe(prompt, max_length=500, num_return_sequences=1)

# Extract fake answers from the generated text
generated_text = generated_texts[0]['generated_text']

# ‍print(f"The generated text is: \n {generated_text.split('*')[1].split('</think>')}")
# print(generated_texts)


# print(generated_text)
print(f"The generated text is: \n {generated_text.split('*')[1]}")





random.randint(0, 4)


def get_essay(topic):
    prompt = f"""
    Generate a piece of text about {topic} message with the following considerations: 
    1- Make sure it doesn't exceed 100 words.
    2- Add a few conflicting view points.
    3- Ensure the text is in clear and fluent English.
    4- Generate the text using a {get_item(perspectives)} perspective.
    5- Please generate the text with a / an {get_item(tones)} tone.
    6- Please generate the text with a / an {get_item(styles)} style.
    *
    """

    # Generate text using the pipeline
    generated_texts = pipe(prompt, max_length=1000, num_return_sequences=1)
    
    # Extract fake answers from the generated text
    generated_text = generated_texts[0]['generated_text']
    
    # print(f"The generated text is: \n {generated_text.split('*')[1]}")

    generated_text = generated_text.split('*')[1]
    

    return generated_text


tqdm.pandas()
df_test["essay"] = df_test["topic"].progress_apply(lambda x : get_essay(x))


df_test['essay'].loc[1]


df_test.drop(columns=["topic"], inplace=True)


df_test.to_csv("/kaggle/working/submission.csv", index=False)

