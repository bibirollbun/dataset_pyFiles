# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import Image
from IPython.display import Markdown, display
from datasets import load_dataset,Dataset
import ast
import kagglehub
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

! pip install accelerate -q
! pip install -i https://pypi.org/simple/ bitsandbytes -q
! pip install peft -q
! pip install trl -q
! pip install git+https://github.com/huggingface/datasets -U -q
! pip install git+https://github.com/huggingface/transformers -U -q

! pip install --upgrade trl 

import os
import torch

import numpy as np
import pandas as pd

from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          AutoConfig,
                          TrainingArguments)

from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from IPython.display import Markdown, display

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
#from peft import LoraConfig,PeftModel


Image("/kaggle/input//example/culture.jpg")


df =  pd.read_csv("/kaggle/input/wikipedia-movie-plot/movie_plot.csv")

df.head(5)



# Count the occurrences of each year
year_counts = df['year'].value_counts().sort_index()
# Plotting
plt.figure(figsize=(8, 6))
year_counts.plot(kind='bar')
# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Count of Movies per Year')
plt.xticks(rotation=90) # Rotate x labels for better readability
# Show the plot
plt.show()


language_counts = df['language'].value_counts()
# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Language Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# Display the pie chart
plt.show()


language_counts = df['country'].value_counts()
# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Language Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# Display the pie chart
plt.show()


model_id = "/kaggle/input/gemma-2/transformers/gemma-2-2b-it/2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,device_map='auto')



def generate_response(model, tokenizer, prompt, device, max_new_tokens=128):
    """
    This function generates a response to a given prompt using a specified model and tokenizer.

    Parameters:
    - model (PreTrainedModel): The machine learning model pre-trained for text generation.
    - tokenizer (PreTrainedTokenizer): A tokenizer for converting text into a format the model understands.
    - prompt (str): The initial text prompt to generate a response for.
    - device (torch.device): The computing device (CPU or GPU) the model should use for calculations.
    - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 128.

    Returns:
    - str: The text generated in response to the prompt.
    """
    # Convert the prompt into a format the model can understand using the tokenizer.
    # The result is also moved to the specified computing device.
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate a response based on the tokenized prompt.
    outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=max_new_tokens)

    # Convert the generated tokens back into readable text.
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract and return the response text. Here, it assumes the response is formatted as "Response: [generated text]".
    response_text = text.split("Response:")[1]
    
    return response_text

instruction = "Generate the plot for an Indian tamil romantic movie reflecting 2015's culture"

template = "Instruction:\n{instruction}\n\nResponse:\n{response}"

prompt = template.format(
    instruction=instruction,
    response="",
)

response_text = generate_response(model, tokenizer,prompt, "cuda",1024)

print("Instruction:",instruction)
print("---------------------")
Markdown(response_text)


# Create Movie plot Instruction 
def create_plot_prompt(row):
  """Generates a plot prompt string based on movie attributes.

  Args:
    row: A pandas Series representing a movie row.

  Returns:
    A string with the plot prompt.

  """
  genre_str = ",".join(ast.literal_eval(row['genre']))
  return f"Generate the plot for an {row['country']} {row['language']} {genre_str} movie reflecting {row['year']}'s culture"

df['instruction'] = df.apply(create_plot_prompt, axis=1)

print("Instruction example:",df['instruction'].tolist()[0])


# Filter movies which do not have detailed plot
df['token_length'] = df['plot'].apply(lambda x: len(tokenizer.encode(x)))
df_filtered = df[(df.token_length>=400) & (df.token_length<2000)]
dataset = Dataset.from_pandas(df_filtered)
print("Training size:",len(df_filtered))


import wandb

# Initialize Weights & Biases (wandb) for experiment tracking.
# If a wandb account exists, it can typically be used by specifying project and entity.
# However, for this example, we're disabling wandb to ignore it by setting mode to "disabled".
wandb.init(mode="disabled")

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

instruction_column = 'instruction'
response_column = "plot"
def formatting_func(example):
    """
    Formats a given example (a dictionary containing question and answer) using the predefined template.
    
    Parameters:
    - example (dict): A dictionary with keys corresponding to the columns of the dataset, such as 'question' and 'answer'.
    
    Returns:
    - list: A list containing a single formatted string that combines the instruction and the response.
    """
    # Add the phrase to verify training success and format the text using the template and the specific example's instruction and response.
    line = template.format(instruction=example[instruction_column], response=example[response_column])
    return [line]

from trl import SFTTrainer
import trl
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        #max_steps=100,
        num_train_epochs=10,  
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func
)


trainer.train()


model_trained=trainer.model.half()
instruction = "Generate the plot for an Indian tamil romantic movie reflecting 2015's culture"

prompt = template.format(
    instruction=instruction,
    response="",
)

device = "cuda"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model_trained.generate(**inputs,temperature = 0.4,do_sample=True,max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


LOCAL_MODEL_DIR = 'output/gemma'

model_trained.save_pretrained(LOCAL_MODEL_DIR, safe_serialization=True) 

MODEL_SLUG = 'gemma-2-movie-plot-generator' # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = '1' # Replace with variation slug.

kagglehub.model_upload(
  handle = f"mechallenge/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
  local_model_dir = LOCAL_MODEL_DIR,
  version_notes = 'Update 2025-01-14')


