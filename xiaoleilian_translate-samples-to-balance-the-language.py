pip install openai


from openai import OpenAI
import os
import copy
from dataclasses import dataclass
import traceback
import numpy as np
import torch
import pandas as pd
import random
from tqdm import tqdm
import re



client = OpenAI(api_key="<api-key>", base_url="https://api.deepseek.com")  # Replace with yours correct api_key and base_url

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)

def translate(text, lang):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are translator, only output translated content."},
            {"role": "user", "content": f"Translate the text below into {lang}: {text}"},
        ],
        stream=False
    )

    return response.choices[0].message.content



df = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')
df


lang_dict = {}
for idx, row in df['language'].value_counts().reset_index().iterrows():
    if row['count'] >= 10:
        lang_dict[row['language']] = row['count']
    else:
        lang_dict[row['language']] = 10

del lang_dict['English']


lang_dict


def is_mostly_english(text):
    total_chars = len(text)
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    # Return True if the proportion of English characters is greater than 80%
    return english_chars / total_chars > 0.8 if total_chars > 0 else False

# Filter samples that meet the criteria
filtered_df = df[
    (df['language'] == 'English') &  # Select rows where the 'language' column is 'English'
    df['prompt'].apply(is_mostly_english) &  # The 'prompt' column contains mostly English characters
    ~df['prompt'].str.contains(r'\btranslate\b', case=False)  # Exclude rows containing the word 'translate' (case insensitive)
]
filtered_df


# 获取语言名称和对应的权重
langs = list(lang_dict.keys())  # ['english', 'france', 'russian', 'chinese']
weights = list(lang_dict.values())  # [0.2, 0.1, 0.1, 0.1]

new_df = pd.DataFrame()
for idx, row in tqdm(filtered_df.copy().iterrows(), total=len(filtered_df)):    
    selected_language = random.choices(langs, weights=weights, k=1)[0]
    # TODO Translate
    p = translate(row['prompt'], selected_language)
    r_a = translate(row['response_a'], selected_language)
    r_b = translate(row['response_b'], selected_language)
    row['prompt'] = p
    row['response_a'] = r_a
    row['response_b'] = r_b
    row['language'] = selected_language

    # append the translated sample
    new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

    if idx > 20: break  # test flag


new_df.to_csv('extra_samples.csv', index=False)

