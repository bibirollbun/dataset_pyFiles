# Title: S.O.M. 1.0 - Santa 2024 Perplexity Puzzle Solver
# Subtitle: Harmonic Reorganization of Words Inspired by T-Física Principles

# Copyright 2025 Emerson Italo Lima da Silva (Tiberius)
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Importing essential libraries
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import permutations
import logging

# General configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Model and tokenizer setup
MODEL_NAME = "gpt2"  # Change to "Gemma 2 9B" or other as per competition requirements
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Load dataset
def load_dataset(input_path):
    logging.info(f"Loading dataset from: {input_path}")
    return pd.read_csv(input_path)

# Evaluate perplexity of a sentence
def calculate_perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
    return np.exp(loss)

# Generate permutations and evaluate perplexity
def optimize_permutation(base_sequence):
    words = base_sequence.split()
    best_permutation = None
    lowest_perplexity = float("inf")
    
    logging.info(f"Generating permutations for sequence: {base_sequence}")
    for perm in permutations(words):
        perm_sentence = " ".join(perm)
        perplexity = calculate_perplexity(perm_sentence)
        if perplexity < lowest_perplexity:
            best_permutation = perm_sentence
            lowest_perplexity = perplexity
    
    logging.info(f"Best permutation: {best_permutation} with perplexity: {lowest_perplexity}")
    return best_permutation

# Process the dataset
def process_data(input_path, output_path):
    dataset = load_dataset(input_path)
    submission = []
    
    logging.info("Processing dataset...")
    for _, row in dataset.iterrows():
        id_, base_sequence = row["id"], row["text"]
        optimized_sequence = optimize_permutation(base_sequence)
        submission.append({"id": id_, "text": optimized_sequence})
    
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(output_path, index=False)
    logging.info(f"Submission file saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    input_path = "./input/sample_submission.csv"  # Adjust path as needed
    output_path = os.path.join(output_dir, "submission.csv")
    
    logging.info("Starting Santa 2024 Perplexity Puzzle Solver...")
    process_data(input_path, output_path)
    logging.info("Processing complete. Ready for submission!")


