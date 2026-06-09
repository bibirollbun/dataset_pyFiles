# 1. Install Deps
!pip install numpy scipy scikit-learn pandas matplotlib seaborn
!pip install unsloth

!pip install trl
!pip install wandb
!pip install deepspeed==0.14.4
!pip install optuna
!pip install openai
!pip install pydantic
!pip install transformers
!pip install torch --upgrade

# Install Flash Attention 2 for softcapping support
import torch
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install flash-attn --no-build-isolation


# Setup
import pandas as pd
from IPython.display import Markdown, display
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import optuna
from transformers import TrainingArguments
from trl import SFTTrainer
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
from transformers import pipeline

import matplotlib.pyplot as plt
import numpy as np
from math import pi
import json
import kagglehub
import wandb


import os

# Set the environment variable for PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "unlock-communication-gemma-2"

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
openai_api_key = user_secrets.get_secret("OPENAI_API_KEY")
wandb_api_key = user_secrets.get_secret("WANDB_API")


os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['HF_TOKEN'] = hf_token

client = OpenAI()
wandb.login(key=wandb_api_key)


urdu_poetry_explanation_path = kagglehub.dataset_download('thee5z/urdu-poetry-explanation')
df = pd.read_csv(f'{urdu_poetry_explanation_path}/sher_explanation_final.csv')
df.sample(n=5)


row = df.sample(n=1).iloc[0]

display(Markdown(f"""
{row['text']}

{row['sher_explanation']}
                 """))


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

x = df.iloc[0]
x

chat = [{
    "role": "user",
    "content": f"""{x['text']}"""
}, 
{
    "role": "assistant",
    "content": f"""{x['sher_explanation']}"""
}]

display(Markdown(tokenizer.apply_chat_template(chat, tokenize=False))) 


df['training_text'] = df.apply(lambda x: tokenizer.apply_chat_template([{
    "role": "user",
    "content": f"""{x['text']}"""
}, 
{
    "role": "assistant",
    "content": f"""{x['sher_explanation']}"""
}], tokenize=False, add_generation_prompt=False), axis=1)

df.head()


dataset = Dataset.from_pandas(df[['training_text']])
dataset


def tokenize_function(samples):
    return tokenizer(samples["training_text"])


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset


training_data = tokenized_dataset.train_test_split(test_size=0.1)
training_data


model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="cuda", torch_dtype=torch.bfloat16)


def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 6])
    num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)

    training_args = TrainingArguments(
        output_dir=f"./optuna_trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_dir=f"./logs_trial_{trial.number}",
        logging_steps=10,
        bf16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=training_data["train"],
        eval_dataset=training_data["test"],
        args=training_args,
        tokenizer=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    return eval_results["eval_loss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

# Get the best hyperparameters
best_hyperparams = study.best_trial.params
print("Best Hyperparameters:", best_hyperparams)


# Make space for the final fine tune
torch.cuda.empty_cache()


training_args = TrainingArguments(
    output_dir="./gemma2-sft-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="wandb",
    learning_rate=4e-5,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    bf16=True,
)

trainer = trl.SFTTrainer(
    model=model,
    train_dataset=training_data["train"],
    eval_dataset=training_data["test"],
    args=training_args,
    tokenizer=tokenizer,
)


trainer.train()


trainer.save_model("sft_model_output")
trainer.tokenizer.save_pretrained("sft_model_output")


testing_data = pd.read_csv(f'{urdu_poetry_explanation_path}/testing_data.csv')
random_row = testing_data.sample(n=1).iloc[0]
random_row

user_instruction = 'اس شعر کو سمجھائیے۔'


def call_openai_api(prompt):
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
    return response.choices[0].message.content


display(Markdown(random_row['text']))
display(Markdown(call_openai_api(f'What is the english transliteration for this: {random_row["text"]}')))


chatgpt_analysis = call_openai_api(f"{user_instruction}\n{random_row['text']}")


from transformers import pipeline

# Load the tokenizer and model
model = AutoModelForCausalLM.from_pretrained("/kaggle/input/gemma2-2b-it-urdu-poetry/transformers/default/1/sft_model_output", device_map="cuda", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma2-2b-it-urdu-poetry/transformers/default/1/sft_model_output")
model.eval()

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16})

messages = [{
    "role": "user",
    "content": f"{user_instruction}\n{random_row['text']}"
}]

outputs = pipe(messages, do_sample=False, max_new_tokens=1000)
outputs


finetune_model_analysis = outputs[0]['generated_text'][1]['content']


html_table = f"""
<table style="width: 100%; table-layout: fixed; word-wrap: break-word; border-collapse: collapse;">
    <tr>
        <th style="border: 1px solid black; padding: 8px;">ChatGPT Analysis</th>
        <th style="border: 1px solid black; padding: 8px;">Finetune Model Analysis</th>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 8px; white-space: pre-wrap;">{chatgpt_analysis}</td>
        <td style="border: 1px solid black; padding: 8px; white-space: pre-wrap;">{finetune_model_analysis}</td>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 8px; white-space: pre-wrap;">{call_openai_api(f"Translate this to english {chatgpt_analysis}")}</td>
        <td style="border: 1px solid black; padding: 8px; white-space: pre-wrap;">{call_openai_api(f"Translate this to english {finetune_model_analysis}")}</td>
    </tr>
</table>
"""

display(Markdown(html_table))


class Evaluation(BaseModel):
    rating: int
    ratingMax: int
    remarks: str

class RubricItem(BaseModel):
    name: str
    description: str

def evaluate_analysis(sher: str, analysis: str, rubric: List[RubricItem]):
    results = {}
    for item in rubric:
        response =  client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are a poetry analysis evaluator providing feedback strictly based on a given rubric. Keep the remarks concise and to the point."},
                {"role": "user", "content": f"Given the Urdu sher: '{sher}', rate the following analysis based on the rubric described: {item.description}. Analysis: {analysis}"}
            ],
            response_format=Evaluation  # Passing Pydantic directly here
        )
        evaluation = response.choices[0].message.parsed # Parsing directly into Pydantic model
        results[item.name] = evaluation.model_dump_json()

    # Converting values to JSON objects
    converted_results = {}

    for key, value in results.items():
        try:
            converted_results[key] = json.loads(value)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for key: {key}")

    return converted_results

rubric_items = [
    RubricItem(
        name="Readability and Flow",
        description=(
            "The 'Readability and Flow' (ratingMax = 7 points) rubric item evaluates the extent to which the analysis is presented in a natural, flowing style that engages the reader. "
            "Consider whether it feels like a thoughtful, human explanation rather than a rigid list of bullet points. "
            "The analysis should avoid sounding AI-generated or mechanical. If the explanation feels generic or uninspired, the score should be lower. "
            "Higher scores should be given for a human-like tone that is reflective, conversational, and engaging."
            "If the answer is not in Urdu script, the score should be 0."
        )
    ),
    RubricItem(
        name="Structure & Flow of the Analysis",
        description=(
            "The 'Structure & Flow of the Analysis' (ratingMax = 3 points) rubric item assesses the overall organization and logical progression of the analysis. "
            "Ensure that the points flow smoothly from one aspect to another without redundancy. "
            "The analysis should focus on providing meaningful insights rather than repeating the same ideas. "
            "The tone should be engaging and reflective of the emotional and artistic depth of the sher."
        )
    ),
    RubricItem(
        name="Structural Understanding",
        description=(
            "The 'Structural Understanding' Structural Understanding (ratingMax = 3 points) rubric item examines how well the analysis explores the relationship between the two lines of the sher: the Misra-e-Oola and Misra-e-Sani. "
            "Determine if the analysis clearly explains how the second line resolves, deepens, or contrasts with the first. "
            "Ensure the analysis addresses elements of meter (behr), rhyme (qaafiya), and refrain (radeef) accurately and comprehensively."
        )
    ),
    RubricItem(
        name="Interpretation & Meaning",
        description=(
            "The 'Interpretation & Meaning' Structural Understanding (ratingMax = 3 points) rubric item evaluates whether the analysis provides both surface-level and deeper interpretations of the sher. "
            "Consider if the analysis explores cultural, historical, or mystical contexts where relevant. "
            "The analysis should also connect the poet’s ideas to broader philosophical themes and insights for a richer understanding."
        )
    ),
    RubricItem(
        name="Depth of Language Analysis",
        description=(
            "The 'Depth of Language Analysis' Structural Understanding (ratingMax = 3 points) rubric item assesses the depth of analysis regarding the poet’s lexical choices, focusing on the balance between simplicity and sophistication. "
            "The analysis should discuss syntax and word placement in terms of their effect on meaning and flow. "
            "It should also comment on the economy of language and how individual words contribute to the sher’s impact."
        )
    ),
    RubricItem(
        name="Imagery & Poetic Devices",
        description=(
            "The 'Imagery & Poetic Devices' Structural Understanding (ratingMax = 3 points) rubric item analyzes the exploration of sensory elements, metaphors, contrasts, and both abstract and visual imagery within the analysis. "
            "Identify whether key literary devices like similes (tashbih), metaphors (istiara), and wordplay (tajnis) are effectively discussed. "
            "The analysis should also clearly articulate the cultural and symbolic meanings of words and phrases where applicable."
        )
    )
]


results = evaluate_analysis({random_row['text']}, finetune_model_analysis, rubric_items)
print(results)


# Prepare data for radar chart
categories = list(results.keys())
scores = [x['rating'] for x in list(results.values())]
max_scores = [x['ratingMax'] for x in list(results.values())]

# Extract numerical values from category labels for normalization
normalized_scores = [score / max_score for score, max_score in zip(scores, max_scores)]

# Setting up the radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

normalized_scores += normalized_scores[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})

ax.fill(angles, normalized_scores, color='blue', alpha=0.25)
ax.plot(angles, normalized_scores, color='blue', linewidth=2)
ax.set_yticklabels([])

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, rotation=30)

# Adding the score labels
for i, score in enumerate(scores):
    angle_rad = angles[i]
    ax.text(angle_rad, normalized_scores[i] + 0.1, f'{score}/{max_scores[i]}', fontsize=10)

plt.title("Poetry Analysis Evaluation Results")
plt.show()



model = AutoModelForCausalLM.from_pretrained("/kaggle/input/gemma2-2b-it-urdu-poetry/transformers/default/1/sft_model_output", device_map="cuda", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma2-2b-it-urdu-poetry/transformers/default/1/sft_model_output")

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.bfloat16})

messages = [{
    "role": "user",
    "content": 'اس شعر کو سمجھائیے۔\nدل کا دکھ جانا تو دل کا مسئلہ ہے پر ہمیں \nاس کا ہنس دینا ہمارے حال پر اچھا لگا \n'
}]

outputs = pipe(messages, do_sample=False, max_new_tokens=1000)
outputs

