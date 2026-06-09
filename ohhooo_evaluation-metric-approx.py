import os
import numpy as np
import pandas as pd
import re
import json


import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings('ignore')

import subprocess
import sys

!pip install /kaggle/input/python-ku2/lingua_language_detector-2.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
from lingua import Language, LanguageDetectorBuilder

import difflib
from itertools import combinations


import platform
print(platform.system())  # 查看操作系统类型
print(platform.version())  # 查看操作系统版本


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2")
model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2",
    device_map="auto",
    trust_remote_code=True,
)
model.config.pad_token_id = model.config.eos_token_id


judge_model_list = [
    "/kaggle/input/phi/transformers/2/1",
    "/kaggle/input/qwen2.5/transformers/3b-instruct/1",
    "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2/",    
]

models = [AutoModelForCausalLM.from_pretrained(model_name, device_map='cpu', torch_dtype=torch.bfloat16) for model_name in judge_model_list]
tokenizers = [AutoTokenizer.from_pretrained(model_name) for model_name in judge_model_list]


def essay_create(test_data):
    text = """You are a writing expert. Your task is to generate three essays for each topic provided by the user.And you need to ensure the integrity of your paper.
## Generate three essays for each topic:
## Note:
* The result should be output in JSON format, following the specified structure
## Notice1:You need to ensure the integrity of your output paper.
## Notice2:Each essay should be approximately 120 words
## Output in JSON format
```json
{
    "topic-index": <topic index, formatted as n, where n is an integer starting from 1>,
    "topic": <topic name>,
    "essay-number": <essay index, formatted as n, where n is an integer starting from 1>,
    "essay": <essay content>
}
```
"""
    input_title = tokenizer(text, return_tensors="pt")
    input_title = {k: v.to('cuda') for k, v in input_title.items()}
    outputs1 = model.generate(**input_title,max_new_tokens=150)
    output_text1 = tokenizer.decode(outputs1[0][len(input_title['input_ids'][0]):])
    print(output_text1)
    essays = []
    for i, row in test_data.iterrows():
        content = f"""topic{i+1}:{row['topic']}"""
        print(content)
        input_text = tokenizer(content, return_tensors="pt")
        input_text = {k: v.to('cuda') for k, v in input_text.items()}
        outputs = model.generate(**input_text, max_new_tokens=200)
        output_text = tokenizer.decode(outputs[0][len(input_text['input_ids'][0]):])
        print(output_text)
        essays.append(output_text)
    return essays


def index_create():
    text = """## Propose general metrics for evaluating essays' aspects:
* Metrics should cover various aspects of essay evaluation
## Note:
* The output should be in JSON format, following the specified structure

## Output in JSON format
```json
{
    "metric-index": [
        Metric Index1,
        Metric Index2,
        ...
    ],
    "metric-name": [
        Metric Name1,
        Metric Name2,
        ...
    ]
}
```
"""
    input_title = tokenizer(text, return_tensors="pt")
    input_title = {k: v.to('cuda') for k, v in input_title.items()}
    outputs = model.generate(**input_title,max_new_tokens=200)
    output_text = tokenizer.decode(outputs[0][len(input_title['input_ids'][0]):])
    print(output_text)
    match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if match:
        json_content = match.group()
    json_content = re.sub(r'[\x00-\x1F\x7F]', '', json_content)
    data = json.loads(json_content)

    metric_index = data.get("metric-index", [])
    metric_name = data.get("metric-name", [])
    return metric_index,metric_name


def index_secondary_create(metric):
    text = f"""## Propose sub-indicators to evaluate various aspects of this paper indicator <{metric}>:
* Sub-indicators should cover various aspects of this paper indicator <{metric}>.
## Note:
* The output should be in JSON format and follow the specified structure.
Sub-indicators:""" + """
```json
{
    "metric-index": [
        Metric Index1,
        Metric Index2,
        ...
    ],
    "metric-name": [
        Metric Name1,
        Metric Name2,
        ...
    ]
}
```
"""
    input_title = tokenizer(text, return_tensors="pt")
    input_title = {k: v.to('cuda') for k, v in input_title.items()}
    outputs = model.generate(**input_title,max_new_tokens=200)
    output_text = tokenizer.decode(outputs[0][len(input_title['input_ids'][0]):])
    print(output_text)
    match = re.search(r'\{.*\}', output_text, re.DOTALL)
    if match:
        json_content = match.group()
    json_content = re.sub(r'[^\x20-\x7E]', '', json_content)
    data = json.loads(json_content)

    metric_index = data.get("metric-index", [])
    metric_name = data.get("metric-name", [])
    return metric_index,metric_name


def essay_change_high_metric(essays, metric):
    high_metric_score_essays = []
    
    for i, essay in enumerate(essays, 1):
        title1 = f"""You are an essay grading expert. Your task is to score and modify the essay provided by the user based on the <{metric}> metric. 
        ## Score and modify the essay based on the metric: 
        * Analyze the content of the essay 
        * Evaluate the essay based on the metric and generate the corresponding score 
        * Modify the original paper to improve its metric score, thereby creating a paper with a high metric score 
        ## Notice1:You need to ensure the integrity of your output paper. 
        ## Notice2:Each modified essay should be approximately 150 words 
        ## Notice3:Output in JSON format""" + """
```json
{
    "Original paper metric score": "<Score from 0 to 9>",
    "Modified paper metric score": "<Score from 0 to 9>",
    "modified essay":<modified essay>
}```
## Notice4:The result should be in JSON format only 
## Notice5:Only output the JSON Essay as follows: """+ f"""
{essay}"""
        
        input_title1 = tokenizer(title1, return_tensors="pt")
        input_title1 = {k: v.to('cuda') for k, v in input_title1.items()}
        outputs1 = model.generate(**input_title1, max_new_tokens=700)
        output_text1 = tokenizer.decode(outputs1[0][len(input_title1['input_ids'][0]):])
        print(f'high_scoring_essay{i}')
        match = re.search(r'\{.*\}', output_text1, re.DOTALL)
        if match:
            json_content = match.group()

        json_content = re.sub(r'[\x00-\x1F\x7F]', '', json_content)

        # json_content = json_content.replace("\n", "").replace("\r", "")
        print(json_content)
        data = json.loads(json_content)
        high_metric_score_essays.append(data.get("modified essay", []))
    return high_metric_score_essays

def essay_change_low_metric(essays, metric):
    low_metric_score_essays = []
    
    for i, essay in enumerate(essays, 1):
        title1 = f"""You are an essay grading expert. Your task is to score and modify the essay provided by the user based on the <{metric}> metric. 
        ## Score and modify the essay based on the metric: 
        * Analyze the content of the essay 
        * Evaluate the essay based on the metric and generate the corresponding score 
        * Modify the original paper to lower its metric score, thereby creating a paper with a low metric score 
        ## Notice1:You need to ensure the integrity of your output paper. 
        ## Notice2:Each modified essay should be approximately 150 words 
        ## Notice3:Output in JSON format""" + """
```json
{
    "Original paper metric score": "<Score from 0 to 9>",
    "Modified paper metric score": "<Score from 0 to 9>",
    "modified essay":<modified essay>
}```
## Notice4:The result should be in JSON format only 
## Notice5:Only output the JSON Essay as follows: """+ f"""
{essay}"""
        
        input_title1 = tokenizer(title1, return_tensors="pt")
        input_title1 = {k: v.to('cuda') for k, v in input_title1.items()}
        outputs1 = model.generate(**input_title1, max_new_tokens=700)
        output_text1 = tokenizer.decode(outputs1[0][len(input_title1['input_ids'][0]):])
        print(f'low_scoring_essay{i}')

        match = re.search(r'\{.*\}', output_text1, re.DOTALL)
        if match:
            json_content = match.group()
        json_content = re.sub(r'[\x00-\x1F\x7F]', '', json_content)
        print(json_content)
        data = json.loads(json_content)
        low_metric_score_essays.append(data.get("modified essay", []))
    return low_metric_score_essays


def llm_judge(prompt, response, criteria, model, tokenizer):
    """
    Evaluate a list of responses with scores (0-9) using a Hugging Face Transformer model.

    Args:
        prompt (str): The initial task or question given to the respondents.
        response (str): Response to evaluate.
        criteria (str): Evaluation criteria to judge the response.
        model: Huggingface model to use.
        tokenizer: Huggingface tokenizer to use.

    Returns:
        score: Score for the response.
    """
    # Build the evaluation prompt
    evaluation_prompt = f"""
You are an expert judge scoring responses to the following prompt:

Prompt: {prompt}

Evaluation Criteria: {criteria}

Provide a score between 0 and 9 (inclusive) for the response. Do not provide any explanation.

Here is the response to evaluate:
"""
    evaluation_prompt += f"\nResponse: {response}\nScore:"

    # Tokenize the input
    inputs = tokenizer(evaluation_prompt, return_tensors="pt")
    inputs = {k: v.to('cpu') for k, v in inputs.items()}

    # Generate output
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        num_return_sequences=1,
        temperature=0,
        do_sample=False
    )

    # Decode the output
    evaluation_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        score_line = evaluation_output.split(f"Response:")[1].split("\nScore:")[1].strip()
        score = float(score_line.split()[0])  # Extract the numeric score
    except Exception:
        score = 0

    return score


judging_criteria = "Clarity, relevance to the topic, and strength of the argument."


def essay_score(path):
    submission_df = pd.read_csv(path)
    submission_df.head()
    for model in models:
        model.config.pad_token_id = model.config.eos_token_id
    avg_qs = []
    avg_variances = []
    
    for i, row in submission_df.iterrows():
        task_prompt = f"Write an essay on the topic {row['topic']}"
        results = [llm_judge(task_prompt, row['essay'], 
        judging_criteria, model, tokenizer) for (model, tokenizer) in zip(models, tokenizers)]
        avg_qs.append(np.mean(results))
        avg_variances.append(np.var(results))
    submission_df['avg_q'] = avg_qs[i]
    submission_df['avg_variance'] = avg_variances[i]
    print(submission_df)
    # Build the language detector
    detector = LanguageDetectorBuilder.from_all_languages().build()
    english_confidence = []
    
    for i, row in submission_df.iterrows():
        # Compute language confidence values
        results = detector.compute_language_confidence_values(row['essay'])
        confidence = next((result.value for result in results if result.language == Language.ENGLISH), 0.0)
        english_confidence.append(confidence)
    submission_df['avg_e'] = english_confidence
    essays = submission_df['essay'].values
    similarities = [
            difflib.SequenceMatcher(a=essay1, b=essay2).ratio() for essay1, essay2 in combinations(essays, 2)
        ]
    avg_s = sum(similarities) / len(similarities)
    MIN_S = 0.2
    avg_s_clipped = max(avg_s, MIN_S)
    MAX_Q = 9
    final_score = (submission_df['avg_variance'].mean() / (MAX_Q - submission_df['avg_q'].mean())) * (submission_df['avg_e'].mean() / avg_s_clipped)
    print(final_score)
    return final_score


max_score = 0


test_data = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
test_data.head()
for index, topic in test_data['topic'].items():
    result = topic

metric_index,metric_name = index_create()
print('mertic_index:',metric_index)
print('mertic_name:',metric_name)

essays = []
essays = essay_create(test_data)

test_data['essay'] = essays
test_data.head()


j = 0
all_max_metric = []
while(max_score<20 or j>3):
    j = j+1
    for i in range(len(metric_name)):
        high_change_essays_data = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv', nrows=500)
        low_change_essays_data = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv', nrows=500)
        
        high_change_essays = essay_change_high_metric(essays, metric_name[i])
        low_change_essays = essay_change_low_metric(essays, metric_name[i])
        high_change_essays_data['essay'] = high_change_essays
        low_change_essays_data['essay'] = low_change_essays
        print(high_change_essays_data)
        print(low_change_essays_data)
        high_change_essays_data.to_csv('high_metric_submission.csv', index=False)
        low_change_essays_data.to_csv('low_metric_submission.csv', index=False)
        
        high_metric_score = essay_score('/kaggle/working/high_metric_submission.csv')
        low_metric_score = essay_score('/kaggle/working/low_metric_submission.csv')
        if high_metric_score > max_score:
            max_metric = (metric_name[i],"high")
            max_score = high_metric_score
        elif low_metric_score > max_score:
            max_metric = (metric_name[i],"low")
            max_score = low_metric_score
    print('max_metric',max_metric,'max_score:',max_score)
    all_max_metric.append(max_metric)
    metric_name = index_secondary_create(max_metric[0])
print(all_max_metric)


data = pd.read_csv('/kaggle/input/llms-you-cant-please-them-all/test.csv')
data.head()
for index, topic in data['topic'].items():
    result = topic

essays = []
essays = essay_create(data)

for i in range(len(all_max_metric)/2):
    if all_max_metric[i*2+1] == 'high':
        essays = essay_change_high_metric(essays,all_max_metric[i*2])
    else:
        essays = essay_change_low_metric(essays,all_max_metric[i*2])

data['essay'] = essays
data.head()

data = data.drop('topic', axis=1)
data.to_csv('submission.csv', index=False)

