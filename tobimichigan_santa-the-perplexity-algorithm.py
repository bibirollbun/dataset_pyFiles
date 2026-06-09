import gc
import os
from math import exp
from collections import Counter
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAM_LIMIT = 15 * 1024  # in MB
VRAM_LIMIT = 10 * 1024  # in MB

# Memory management functions
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Model class with memory-efficient features
class PerplexityCalculator:
    def __init__(self, model_path, load_in_8bit=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32,
            device_map="auto",
        )
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        self.model.eval()

    def calculate_perplexity(self, texts):
        perplexities = []
        for text in texts:
            tokenized = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512,
                add_special_tokens=True
            )
            tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
            with torch.no_grad():
                outputs = self.model(**tokenized)
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = tokenized['input_ids'][..., 1:].contiguous()
                loss = self.loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                perplexities.append(exp(loss.mean().item()))
            clear_memory()  # Clear memory after processing each text
        return perplexities

# Data Preprocessing and Augmentation
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].str.strip()
    return data

# Feature Engineering: Custom Permutation Logic
def permute_words(text):
    words = text.split()
    permutations = sorted(words, key=lambda w: len(w))
    return ' '.join(permutations)

# Training and Evaluation
def train_model(solution_path, model_path):
    # Load and preprocess solution data
    solution = preprocess_data(solution_path)
    
    # Permute words in the text column
    solution['text'] = solution['text'].apply(permute_words)
    
    # Initialize Perplexity Calculator
    scorer = PerplexityCalculator(model_path)
    
    # Compute Perplexity
    solution_texts = solution['text'].tolist()
    perplexities = scorer.calculate_perplexity(solution_texts)
    
    # Add perplexities to the DataFrame
    solution['perplexity'] = perplexities
    
    # Save submission to the desired path
    submission_path = '/kaggle/working/submission.csv'
    solution.to_csv(submission_path, index=False)
    
    print(f"Submission file saved to {submission_path}")
    print(f"Average Perplexity: {np.mean(perplexities):.4f}")
    return perplexities

# Visualization
def plot_metrics(perplexities):
    plt.figure(figsize=(10, 6))
    plt.plot(perplexities, label="Perplexity", marker="o")
    plt.xlabel("Sample Index")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Across Samples")
    plt.legend()
    plt.grid()
    plt.savefig('/kaggle/working/perplexity_plot.png')
    plt.show()

# Main Function
def main():
    # Paths
    solution_path = '/kaggle/input/santa-2024/sample_submission.csv'
    model_path = '/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2'
    
    # Train and Evaluate
    perplexities = train_model(solution_path, model_path)
    
    # Visualize Metrics
    plot_metrics(perplexities)

    # Clear final memory
    clear_memory()

if __name__ == "__main__":
    main()




