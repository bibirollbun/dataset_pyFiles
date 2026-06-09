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


# Install necessary library
!pip install transformers

# Import required libraries
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

# Set up the device (GPU or CPU)
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device_type}")

# Load the GPT-2 tokenizer and model
model_identifier = "gpt2"  # Alternative: "distilgpt2" for faster runtime
text_tokenizer = GPT2Tokenizer.from_pretrained(model_identifier)
language_model = GPT2LMHeadModel.from_pretrained(model_identifier).to(device_type)
language_model.eval()  # Set the model to evaluation mode

# Function to compute perplexity for a given input text
def compute_perplexity(input_text):
    tokenized_input = text_tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=1000
    ).to(device_type)
    with torch.no_grad():
        model_output = language_model(**tokenized_input, labels=tokenized_input["input_ids"])
    calculated_loss = model_output.loss.item()
    perplexity_score = torch.exp(torch.tensor(calculated_loss))
    return perplexity_score.item()

# Function to optimize text using random word reordering
def text_optimization(original_text, beam_size=10, max_attempts=20):
    word_list = original_text.split()
    optimized_sequence = original_text
    lowest_perplexity = compute_perplexity(original_text)

    for _ in range(max_attempts):
        # Randomly reorder the words
        random.shuffle(word_list)
        candidate_sequence = " ".join(word_list)
        candidate_perplexity = compute_perplexity(candidate_sequence)

        # Update the optimized sequence if the perplexity is lower
        if candidate_perplexity < lowest_perplexity:
            optimized_sequence = candidate_sequence
            lowest_perplexity = candidate_perplexity

    return optimized_sequence

# Load the sample data
submission_file = "/kaggle/input/santa-2024/sample_submission.csv"
submission_data = pd.read_csv(submission_file)

# Function to process and optimize all rows in the dataset
def optimize_texts_in_dataset(dataframe, beam_size=3, max_attempts=10):
    updated_texts = []
    for index, row in dataframe.iterrows():
        text_entry = row["text"]
        refined_text = text_optimization(
            text_entry, beam_size=beam_size, max_attempts=max_attempts
        )
        updated_texts.append(refined_text)

        # Show progress every 10 rows
        if (index + 1) % 10 == 0:
            print(f"Optimized {index + 1}/{len(dataframe)} entries")
    
    return updated_texts

# Apply the optimization function and save the updated data
submission_data["text"] = optimize_texts_in_dataset(
    submission_data, beam_size=3, max_attempts=5
)
submission_data.to_csv("submission.csv", index=False)
print("Optimized submission saved as submission.csv")





