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


import os
import ast
import copy
import json
import math
import torch
import random
import warnings
import itertools
import functools
import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
%matplotlib inline

# from tqdm.notebook import tqdm

from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer


warnings.filterwarnings('ignore')
ARC_COLORMAP = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF']
)
ARC_NORM = colors.Normalize(vmin=0, vmax=10)


def show_grid(grid, title=None, figsize=None):
    if not figsize:
        figsize = (len(grid[0]) * 0.6, len(grid) * 0.6)
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap=ARC_COLORMAP, norm=ARC_NORM)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if title:
        plt.title(title, fontsize=12)
    plt.show()


# Define paths
def get_path(name):
    return f'/kaggle/input/arc-prize-2025/{name}' if os.path.exists(f'/kaggle/input/arc-prize-2025/{name}') else name


# Load data files
training_solutions = json.load(open(get_path('arc-agi_training_solutions.json')))
evaluation_solutions = json.load(open(get_path('arc-agi_evaluation_solutions.json')))
evaluation_challenges = json.load(open(get_path('arc-agi_evaluation_challenges.json')))
sample_submission = json.load(open(get_path('sample_submission.json')))
training_challenges = json.load(open(get_path('arc-agi_training_challenges.json')))
test_challenges = json.load(open(get_path('arc-agi_test_challenges.json')))


def visualize_task(task_data, task_solutions=None, title="ARC Task", figsize=(12, 6)):
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    has_solution = task_solutions is not None

    num_train = len(train_examples)
    num_test = len(test_examples)
    cols = num_train + num_test

    fig, axs = plt.subplots(2, cols, figsize=figsize)
    plt.suptitle(title, fontsize=16)

    for idx, example in enumerate(train_examples):
        axs[0, idx].imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, idx].set_title("Train Input")
        axs[0, idx].axis('off')
        axs[1, idx].imshow(example['output'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[1, idx].set_title("Train Output")
        axs[1, idx].axis('off')

    for idx, example in enumerate(test_examples):
        axs[0, num_train + idx].imshow(example['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, num_train + idx].set_title("Test Input")
        axs[0, num_train + idx].axis('off')

        if has_solution:
            axs[1, num_train + idx].imshow(task_solutions[idx], cmap=ARC_COLORMAP, norm=ARC_NORM)
            axs[1, num_train + idx].set_title("Test Output")
        else:
            axs[1, num_train + idx].set_title("Test Output: ?")
        axs[1, num_train + idx].axis('off')


# test_challenges


# visualize_task(task_data)


# Loading the Qwen model
try:
    import kagglehub
    model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/1.7b")
except:
    model_name = "Qwen/Qwen-7B"


# Model loading and setup
try:
    import kagglehub
    model_name = kagglehub.model_download("qwen-lm/qwen-3/transformers/1.7b")
    print(f"Model downloaded from Kaggle Hub: {model_name}")
except ImportError:
    print("Kaggle Hub not available.  Make sure you have kaggle installed and are in a kaggle environment with internet access.")
    model_name = "Qwen/Qwen-7B"  # or any other appropriate Qwen model you have access to
except Exception as e:
     print(f"Error downloading from Kaggle Hub: {e}.  Using a local or alternative model.")
     model_name = "Qwen/Qwen-7B" # Replace this if needed to a suitable model
     print(f"Trying model {model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
).eval()

if device == "cuda":
    model = model.to(device)


class QwenChatbot:
    def __init__(self, model, tokenizer, max_input_tokens=3000):
        self.tokenizer = tokenizer
        self.model = model
        self.history = []
        self.max_input_tokens = max_input_tokens

    def generate_response(self, user_input, max_tokens=64):
        self.history.append({"role": "user", "content": user_input})

        # Try generating with the full history, then truncate if necessary
        while True:
            messages = self.history + [{"role": "user", "content": user_input}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            if inputs.input_ids.shape[-1] <= self.max_input_tokens:
                break  # Good to go
            elif len(self.history) > 2:
                self.history = self.history[2:]  # Truncate oldest user-assistant pair
            else:
                print("⚠️ Unable to fit prompt within token limit even after truncation.")
                return "", ""

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)  # Qwen-specific token
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

        self.history.append({"role": "assistant", "content": response})
        return thinking_content, response


def is_valid_grid(grid):
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    row_length = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != row_length:
            return False
        if not all(isinstance(cell, int) for cell in row):
            return False
    return True
# --- Demonstration and ARC Task Integration ---

# Init chatbot
chatbot = QwenChatbot(model, tokenizer)

# Challenge solving function
def solve_arc_challenge(challenge, chatbot):
    predictions = []

    # Prebuild train prompt
    train_prompt = "Solve the following abstract reasoning challenge:\n\n"
    for i, example in enumerate(challenge['train']):
        train_prompt += f"Input {i + 1}:\n{example['input']}\nOutput {i + 1}:\n{example['output']}\n\n"

    for test_input in challenge['test']:
        prompt = train_prompt + f"Now, predict the output for the following test input:\n{test_input['input']}\nOutput:\n"

        try:
            _, arc_response = chatbot.generate_response(prompt, max_tokens=64)
            arc_response = arc_response.replace(" ", "").replace("\n", "")
            start, end = arc_response.find("[["), arc_response.rfind("]]")
            if start != -1 and end != -1:
                predicted_grid = ast.literal_eval(arc_response[start:end + 2])
                if not is_valid_grid(predicted_grid):
                    raise ValueError("Invalid grid structure")
            else:
                raise ValueError("No valid grid found in response")
        except Exception as e:
            print(f"⚠️ Failed to parse output: {e}")
            predicted_grid = [[0]]  # Safer fallback grid

        predictions.append(predicted_grid)
    return predictions


# # Evaluation loop
# def create_submission(evaluation_challenges, chatbot):
#     submission = {}
#     for task_id, challenge in tqdm(evaluation_challenges.items(), desc="Solving ARC tasks"):
#         predictions = solve_arc_challenge(challenge, chatbot)
        
#         # Validate count matches number of test inputs
#         assert len(predictions) == len(challenge['test']), f"Prediction count mismatch in task {task_id}"
        
#         submission[task_id] = [{"attempt_1": p, "attempt_2": p} for p in predictions]
#     return submission


# # Main
# if __name__ == "__main__":
#     submission = create_submission(evaluation_challenges, chatbot)

#     # Final submission validation
#     for task_id, outputs in submission.items():
#         assert isinstance(outputs, list), f"{task_id} outputs not a list"
#         for entry in outputs:
#             assert 'attempt_1' in entry and 'attempt_2' in entry, f"Missing attempts in {task_id}"
#             assert is_valid_grid(entry['attempt_1']), f"Invalid attempt_1 grid in {task_id}"
#             assert is_valid_grid(entry['attempt_2']), f"Invalid attempt_2 grid in {task_id}"

#     with open('submission.json', 'w') as f:
#         json.dump(submission, f)

#     print("✅ Submission saved to `submission.json`.\n")

#     # Show some results
#     print("=== Sample submission preview ===")
#     sample_tasks = list(submission.items())[:3]  # show first 3 tasks only
#     for task_id, outputs in sample_tasks:
#         print(f"\nTask ID: {task_id}")
#         for i, output in enumerate(outputs):
#             print(f"  Test input #{i+1}:")
#             print(f"    Attempt 1 output: {output['attempt_1']}")
#             print(f"    Attempt 2 output: {output['attempt_2']}")


sample_challenge_id = list(training_challenges.keys())[0]  # Get the first challenge ID
challenge = training_challenges[sample_challenge_id]


challenge


# --- Example of integrating with the ARC dataset ---
# Let's try to answer one of the training challenges.
sample_challenge_id = list(training_challenges.keys())[0]  # Get the first challenge ID
challenge = training_challenges[sample_challenge_id]

# Construct a prompt that describes the task and provides examples
prompt = f"This is a test based on images. Your job is to understand patterns in both training examplesSolve the following abstract reasoning challenge.  Here's the challenge:\n\n"
for i, task in enumerate(challenge['train']): # Use examples from the training set to build the prompt
    prompt += f"Input {i+1}:\n"
    prompt += str(task['input']) + "\n"
    prompt += f"Output {i+1}:\n"
    prompt += str(task['output']) + "\n\n"

# Add the test input.  Tell the model to predict this output
prompt += "Now, predict the output for the following test input:\n. \
Make sure to only return the answer similar to the output examples."
prompt += str(challenge['test'][0]['input']) + "\n"
prompt += "Output:\n"  # Ask the model to generate the output

# Generate the response
print("----------------------")
print("Solving ARC Challenge:")
print(f"Challenge ID: {sample_challenge_id}")
# thinking_content, arc_response = chatbot.generate_response(prompt)
# print(f"Thinking: {thinking_content}")
# print(f"Model's Prediction:\n{arc_response}")
print("----------------------")


prompt


# arc_response


challenge


plt.tight_layout()
plt.show()
visualize_task(challenge)










