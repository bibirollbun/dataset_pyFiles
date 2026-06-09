!pip install transformers --quiet


import itertools
import pandas as pd
import random
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device : ", device)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()


def calculate_perplexity(sequence):
    """
    Given a sequence (string) of words, compute its perplexity using GPT-2.
    """
    # Correct variable name: inputs (instead of imputs)
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Correctly reference the 'input_ids' key in inputs
        outputs = model(**inputs, labels=inputs["input_ids"])

    loss = outputs.loss
    perplexity = torch.exp(loss).item()

    return perplexity


# Load the provided data
data = pd.read_csv("/kaggle/input/santa-2024/sample_submission.csv")

# List of dictionaries to store results
results = []

# Number of random permutations to try per row
# Adjust this for speed vs. performance trade-offs
N_RANDOM_PERMUTATIONS = 10

for idx, row in data.iterrows():
    text_id = row["id"]
    base_text = row["text"]

    words = base_text.split()

    # If the sequence is very large, generating too many permutations is impractical.
    # We'll do a small random search:
    best_sequence = base_text
    best_score = float("inf")

    for _ in range(N_RANDOM_PERMUTATIONS):
        # Shuffle words randomly
        shuffled = random.sample(words, len(words))
        candidate = " ".join(shuffled)

        # Compute perplexity
        score = calculate_perplexity(candidate)

        # Update if we found a lower perplexity
        if score < best_score:
            best_score = score
            best_sequence = candidate

    results.append({"id": text_id, "text": best_sequence})
    print(f"Row {text_id} | Best perplexity so far: {best_score:.2f}")


submission_df = pd.DataFrame(results, columns = ["id", "text"])
submission_df.head()


data.head()


submission_df.to_csv("submission.csv", index=False)




