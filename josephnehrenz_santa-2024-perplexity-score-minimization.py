# Install transformers and torch first if not available
!pip install transformers torch --quiet


import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import permutations
import torch
import pandas as pd

def split_into_chunks(tokens, chunk_size=10):
    """Splits a list of tokens into chunks of size `chunk_size`. The last chunk may be smaller."""
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

def find_best_sequence(tokens, model, tokenizer, max_samples):
    """Finds the best sequence with the lowest perplexity for a list of tokens."""
    best_perplexity = float('inf')
    best_sequence = None

    # Test the alphabetically sorted sequence
    sorted_tokens = ' '.join(sorted(tokens))
    input_ids = tokenizer(sorted_tokens, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**input_ids, labels=input_ids['input_ids'])
        log_likelihood = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(log_likelihood)).item()
    if perplexity < best_perplexity:
        best_perplexity = perplexity
        best_sequence = sorted_tokens

    # Generate random permutations of the tokens
    sampled_permutations = [' '.join(random.sample(tokens, len(tokens))) for _ in range(max_samples)]
    for permuted_text in sampled_permutations:
        # Tokenize the permuted sequence
        input_ids = tokenizer(permuted_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = model(**input_ids, labels=input_ids['input_ids'])
            log_likelihood = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(log_likelihood)).item()

        # Update the best sequence if a lower perplexity is found
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_sequence = permuted_text

    return best_sequence, best_perplexity

def calculate_best_sequences(data, model_path, chunk_size=10, max_samples=16):
    results = []

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    for idx, row in data.iterrows():
        text = row['text']  # Assuming 'text' is the column with the text data

        # Tokenize the text and split into chunks
        tokens = text.split()
        chunks = split_into_chunks(tokens, chunk_size)

        # Find the best sequence for each chunk
        best_chunks = []
        total_perplexity = 0  # Track total perplexity across chunks

        for chunk in chunks:
            best_sequence, best_perplexity = find_best_sequence(chunk, model, tokenizer, max_samples)
            best_chunks.append(best_sequence)
            total_perplexity += best_perplexity  # Accumulate perplexity

        # Combine the best sequences of all chunks
        best_sequence = " ".join(best_chunks)

        # Store the ID, the combined best sequence, and average perplexity
        avg_perplexity = total_perplexity / len(chunks)
        results.append({'id': idx, 'text': best_sequence, 'perplexity': avg_perplexity})

    # Convert the results to a DataFrame
    result_df = pd.DataFrame(results)
    return result_df

# Path to your Gemma model
model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"

# Data
data_file = '/kaggle/input/santa-2024/sample_submission.csv'
data = pd.read_csv(data_file)

# Results df
result_df = calculate_best_sequences(data, model_path, chunk_size=10, max_samples=16)

# Output the result as a .csv file
result_df[['id', 'text']].to_csv('/kaggle/working/perplexity_results.csv', index=False)

# Print the table of IDs and perplexity scores
print(result_df[['id', 'perplexity']])

