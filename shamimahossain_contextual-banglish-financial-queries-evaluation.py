import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


!pip install -q -U immutabledict sentencepiece 
!git clone https://github.com/google/gemma_pytorch.git
!mkdir /kaggle/working/gemma/
!mv /kaggle/working/gemma_pytorch/gemma/* /kaggle/working/gemma/


import sys 
sys.path.append("/kaggle/working/gemma_pytorch/") 
from gemma.config import GemmaConfig, get_model_config
from gemma.model import GemmaForCausalLM
from gemma.tokenizer import Tokenizer
import contextlib
import os
import torch


VARIANT = "2b-v2" 
MACHINE_TYPE = "cuda" 
weights_file = os.path.join(weights_dir, "model.ckpt")


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

model_config = get_model_config(VARIANT)
model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")

device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  model.load_weights(weights_file)
  model = model.to(device).eval()


import json

DATASET_PATH = "/kaggle/input/bangla-eval-data/synthetic_data.json"  # Path to your dataset
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

def build_prompt(query: str) -> str:
    """
    Constructs a structured prompt for financial FAQ use case.
    """
    return (
        "You are a financial assistant trained to answer user queries related to financial transactions, "
        "mobile payments, and services for a financial app called Bkash in Bangladesh. Your answers should be accurate and concise.\n"
        + USER_CHAT_TEMPLATE.format(prompt=query)
        + "<start_of_turn>model\n"
    )

# Evaluation function
def evaluate_dataset(dataset):
    results = []

    for entry in dataset:
        query = entry["query"]
        expected_answer = entry["expected_answer"]

        # Build a financial FAQ-specific prompt
        prompt = build_prompt(query)

        # Generate model response
        generated_output = model.generate(
            prompt,
            device=device,
            output_len=100  # Adjust output length as needed
        )

        # Extract and clean up the generated response
        generated_response = generated_output.strip()

        # Store result
        result = {
            "query": query,
            "expected_answer": expected_answer,
            "generated_response": generated_response
        }
        results.append(result)

    return results




evaluation_results = evaluate_dataset(dataset)


OUTPUT_PATH = "/kaggle/working/evaluation_results.json"
with open(OUTPUT_PATH, "w") as f:
    json.dump(evaluation_results, f, indent=4)

print(f"Evaluation results saved to {OUTPUT_PATH}")


!pip install rouge


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def calculate_scores(results):
    """
    Calculate BLEU and ROUGE scores for generated responses.

    Args:
        results (list): List of dictionaries, each containing:
                        - 'query': The user query
                        - 'expected_answer': The ground truth response
                        - 'generated_response': The model's generated response

    Returns:
        list: A list of dictionaries with BLEU and ROUGE scores for each query.
    """
    rouge = Rouge()
    smoothing_function = SmoothingFunction().method1  # Smoothing for BLEU
    scores = []

    for result in results:
        expected = result["expected_answer"]
        generated = result["generated_response"]

        # Calculate BLEU Score
        bleu_score = sentence_bleu(
            [expected.split()], 
            generated.split(), 
            smoothing_function=smoothing_function
        )

        # Calculate ROUGE Scores
        rouge_scores = rouge.get_scores(generated, expected, avg=True)

        # Store scores
        result_with_scores = {
            "query": result["query"],
            "expected_answer": expected,
            "generated_response": generated,
            "bleu_score": bleu_score,
            "rouge_1": rouge_scores["rouge-1"]["f"],
            "rouge_2": rouge_scores["rouge-2"]["f"],
            "rouge_l": rouge_scores["rouge-l"]["f"]
        }
        scores.append(result_with_scores)

    return scores



# Load evaluation results (after inference)
with open("/kaggle/working/evaluation_results.json", "r") as f:
    evaluation_results = json.load(f)

# Calculate scores
scores = calculate_scores(evaluation_results)

# Save scores to a JSON file
SCORES_OUTPUT_PATH = "/kaggle/working/evaluation_scores.json"
with open(SCORES_OUTPUT_PATH, "w") as f:
    json.dump(scores, f, indent=4)

print(f"Scores saved to {SCORES_OUTPUT_PATH}")



import matplotlib.pyplot as plt
import pandas as pd

def visualize_scores(scores):
    """
    Visualize BLEU and ROUGE scores for the evaluated dataset.

    Args:
        scores (list): List of dictionaries containing:
                       - 'query': The user query
                       - 'bleu_score': BLEU score
                       - 'rouge_1': ROUGE-1 score
                       - 'rouge_2': ROUGE-2 score
                       - 'rouge_l': ROUGE-L score
    """
    # Convert scores to a DataFrame for easier plotting
    df = pd.DataFrame(scores)

    # Sort by BLEU score for better visualization
    df = df.sort_values(by="bleu_score", ascending=False)

    # Plot BLEU Scores
    plt.figure(figsize=(12, 6))
    plt.bar(df["query"], df["bleu_score"], label="BLEU Score", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Query")
    plt.ylabel("Score")
    plt.title("BLEU Scores for Queries")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot ROUGE Scores
    plt.figure(figsize=(12, 6))
    bar_width = 0.2  # Width for grouped bar charts
    index = range(len(df))

    plt.bar(index, df["rouge_1"], bar_width, label="ROUGE-1", alpha=0.7)
    plt.bar([i + bar_width for i in index], df["rouge_2"], bar_width, label="ROUGE-2", alpha=0.7)
    plt.bar([i + 2 * bar_width for i in index], df["rouge_l"], bar_width, label="ROUGE-L", alpha=0.7)

    plt.xticks([i + bar_width for i in index], df["query"], rotation=45, ha="right")
    plt.xlabel("Query")
    plt.ylabel("Score")
    plt.title("ROUGE Scores for Queries")
    plt.legend()
    plt.tight_layout()
    plt.show()



# Load scores (if saved to JSON)
with open("/kaggle/working/evaluation_scores.json", "r") as f:
    scores = json.load(f)

# Visualize scores
visualize_scores(scores)



import matplotlib.pyplot as plt
import numpy as np

def visualize_scores(scores):
    """
    Visualize the overall BLEU and ROUGE scores.

    Args:
        scores (list): List of dictionaries with BLEU and ROUGE scores for each query.
                       Each dictionary should contain keys:
                       - 'bleu_score'
                       - 'rouge_1'
                       - 'rouge_2'
                       - 'rouge_l'
    """
    # Extract scores
    bleu_scores = [entry["bleu_score"] for entry in scores]
    rouge_1_scores = [entry["rouge_1"] for entry in scores]
    rouge_2_scores = [entry["rouge_2"] for entry in scores]
    rouge_l_scores = [entry["rouge_l"] for entry in scores]

    # Calculate averages
    avg_bleu = np.mean(bleu_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_2 = np.mean(rouge_2_scores)
    avg_rouge_l = np.mean(rouge_l_scores)

    # Labels and values for the bar chart
    metrics = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    averages = [avg_bleu, avg_rouge_1, avg_rouge_2, avg_rouge_l]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, averages, color=["blue", "orange", "green", "red"], alpha=0.8)

    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12
        )

    # Title and labels
    plt.title("Overall Model Performance", fontsize=16)
    plt.ylabel("Average Score", fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()



visualize_scores(scores)




