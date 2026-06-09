import random
import numpy as np
import torch
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm.notebook import tqdm
import warnings
import ast


warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """
    Sets the random seeds for reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The following two lines are for ensuring deterministic behavior on GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)
print(f"Random seeds set to {SEED} for reproducibility.")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


ARC_COLORMAP = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])
ARC_NORM = colors.Normalize(vmin=0, vmax=9)


def get_data_path(file_name):
    """Constructs the full path for a data file in the Kaggle environment."""
    return f'/kaggle/input/arc-prize-2025/{file_name}'


def load_json_data(file_name):
    """Loads a JSON file from the competition's dataset."""
    path = get_data_path(file_name)
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {path}. Returning empty dictionary.")
        return {}


def visualize_task(task_data, task_solution=None, task_id=""):
    """
    Plots all input/output pairs for a given ARC task.
    """
    num_train = len(task_data['train'])
    num_test = len(task_data['test'])
    total_pairs = num_train + num_test
    
    fig, axs = plt.subplots(2, total_pairs, figsize=(2.5 * total_pairs, 5.5))
    fig.suptitle(f'Task: {task_id}', fontsize=16, y=1.02)
    
    # Plot training pairs
    for i, pair in enumerate(task_data['train']):
        axs[0, i].imshow(pair['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, i].set_title(f'Train In {i+1}')
        axs[0, i].axis('off')
        
        axs[1, i].imshow(pair['output'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[1, i].set_title(f'Train Out {i+1}')
        axs[1, i].axis('off')

    # Plot test pairs
    for i, pair in enumerate(task_data['test']):
        ax_col = num_train + i
        axs[0, ax_col].imshow(pair['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0, ax_col].set_title(f'Test In {i+1}')
        axs[0, ax_col].axis('off')

        # If we have the solution, plot it. Otherwise, show a placeholder.
        if task_solution and len(task_solution) > i:
            axs[1, ax_col].imshow(task_solution[i], cmap=ARC_COLORMAP, norm=ARC_NORM)
            axs[1, ax_col].set_title(f'Test Out {i+1}')
        else:
            # Placeholder for unknown output
            placeholder = np.full((3, 3), 10, dtype=int) # Using 10 for a white-like color
            cmap_placeholder = colors.ListedColormap(['#FFFFFF'])
            norm_placeholder = colors.Normalize(vmin=10, vmax=10)
            axs[1, ax_col].imshow(placeholder, cmap=cmap_placeholder, norm=norm_placeholder)
            axs[1, ax_col].text(1, 1, '?', ha='center', va='center', fontsize=20)
            axs[1, ax_col].set_title(f'Test Out {i+1}')
        axs[1, ax_col].axis('off')
        
    plt.tight_layout()
    plt.show()


print("Loading datasets...")
train_challenges = load_json_data('arc-agi_training_challenges.json')
train_solutions = load_json_data('arc-agi_training_solutions.json')
eval_challenges = load_json_data('arc-agi_evaluation_challenges.json')
eval_solutions = load_json_data('arc-agi_evaluation_solutions.json')
# This is the file our final submission will run on. For now, it's a placeholder.
test_challenges = load_json_data('arc-agi_test_challenges.json')
print("Datasets loaded.")


print("\nVisualizing a few sample tasks from the training set:")
sample_task_ids = list(train_challenges.keys())[:3]
for task_id in sample_task_ids:
    visualize_task(train_challenges[task_id], train_solutions.get(task_id), task_id)


def analyze_dataset_stats(challenges):
    """
    Computes statistics about the dataset like grid sizes and color counts.
    """
    stats = []
    for task_id, task in challenges.items():
        for pair in task['train']:
            in_grid, out_grid = np.array(pair['input']), np.array(pair['output'])
            all_colors = np.unique(np.concatenate([in_grid.flatten(), out_grid.flatten()]))
            stats.append({
                'task_id': task_id,
                'in_height': in_grid.shape[0],
                'in_width': in_grid.shape[1],
                'out_height': out_grid.shape[0],
                'out_width': out_grid.shape[1],
                'num_colors': len(all_colors)
            })
    return pd.DataFrame(stats)


print("Analyzing training dataset statistics...")
eda_df = analyze_dataset_stats(train_challenges)


fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('EDA: Grid and Color Distributions in Training Set', fontsize=16)

axes[0].hist(pd.concat([eda_df['in_height'], eda_df['out_height']]), bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Grid Height Distribution')
axes[0].set_xlabel('Height')
axes[0].set_ylabel('Frequency')

axes[1].hist(pd.concat([eda_df['in_width'], eda_df['out_width']]), bins=30, color='salmon', edgecolor='black')
axes[1].set_title('Grid Width Distribution')
axes[1].set_xlabel('Width')

axes[2].hist(eda_df['num_colors'], bins=range(1, 12), color='lightgreen', edgecolor='black', align='left')
axes[2].set_title('Unique Colors per Task Pair')
axes[2].set_xlabel('Number of Colors')
axes[2].set_xticks(range(1, 11))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


print(eda_df[['in_height', 'in_width', 'num_colors']].describe())


class HeuristicSolver:
    """
    A simple solver based on the "copy and resize" heuristic.
    """
    def adjust_grid(self, prototype_grid, target_shape):
        """
        Adjusts a grid to a target shape by center-cropping or zero-padding.
        """
        proto_h, proto_w = len(prototype_grid), len(prototype_grid[0])
        target_h, target_w = target_shape
        
        # If shapes match, no adjustment needed
        if (proto_h, proto_w) == (target_h, target_w):
            return prototype_grid

        # Create a new grid of zeros with the target shape
        new_grid = np.zeros(target_shape, dtype=int)
        
        # Determine the slicing for source and destination
        h_slice_proto = slice(max(0, (proto_h - target_h) // 2), min(proto_h, max(0, (proto_h - target_h) // 2) + target_h))
        w_slice_proto = slice(max(0, (proto_w - target_w) // 2), min(proto_w, max(0, (proto_w - target_w) // 2) + target_w))
        h_slice_new = slice(max(0, (target_h - proto_h) // 2), min(target_h, max(0, (target_h - proto_h) // 2) + proto_h))
        w_slice_new = slice(max(0, (target_w - proto_w) // 2), min(target_w, max(0, (target_w - proto_w) // 2) + proto_w))
        
        # Copy the relevant part of the prototype to the new grid
        new_grid[h_slice_new, w_slice_new] = np.array(prototype_grid)[h_slice_proto, w_slice_proto]
        
        return new_grid.tolist()
    def solve(self, task):
        """
        Generates a prediction for a task.
        It uses the first training output as a prototype for all test cases.
        """
        predictions = []
        train_pairs = task.get("train", [])
        
        # Use the first training output as the prototype. If none, use a default.
        if train_pairs and 'output' in train_pairs[0]:
            prototype = train_pairs[0]['output']
        else:
            prototype = [[0]]
        for test_case in task.get("test", []):
            input_grid = test_case.get("input", [[0]])
            target_shape = (len(input_grid), len(input_grid[0]))
            
            pred_grid = self.adjust_grid(prototype, target_shape)
            predictions.append(pred_grid)
            
        return predictions


heuristic_solver = HeuristicSolver()
sample_task_id = 'a7493b83' # A task where this heuristic might work
sample_task = train_challenges.get(sample_task_id)


if sample_task:
    print(f"Testing Heuristic Solver on task: {sample_task_id}")
    heuristic_predictions = heuristic_solver.solve(sample_task)
    
    # Visualize the first test prediction from the heuristic
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(sample_task['test'][0]['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
    axs[0].set_title("Test Input")
    axs[1].imshow(heuristic_predictions[0], cmap=ARC_COLORMAP, norm=ARC_NORM)
    axs[1].set_title("Heuristic Prediction")
    axs[2].imshow(train_solutions[sample_task_id][0], cmap=ARC_COLORMAP, norm=ARC_NORM)
    axs[2].set_title("Ground Truth")
    for ax in axs: ax.axis('off')
    plt.show()


# --- Installation for gemma_pytorch ---
# We install the required libraries and clone the official gemma_pytorch repository.
!pip install -q -U immutabledict sentencepiece
!git clone https://github.com/google/gemma_pytorch.git
import sys
sys.path.append("/kaggle/working/gemma_pytorch/")



import sys 
# Add the cloned repository to the Python path
sys.path.append("/kaggle/working/gemma_pytorch/") 
from gemma.config import get_model_config
from gemma.model import GemmaForCausalLM
import contextlib
import os
import torch
import re
import ast


VARIANT = "2b-v2" 
MACHINE_TYPE = "cuda" if torch.cuda.is_available() else "cpu" 
model = None

try:
    print(f"Loading Gemma 2 ({VARIANT}) model...")
    weights_dir = '/kaggle/input/gemma-2/pytorch/gemma-2-2b-it/1'
    weights_file = os.path.join(weights_dir, "model.ckpt")
    
    @contextlib.contextmanager
    def _set_default_tensor_type(dtype: torch.dtype):
      torch.set_default_dtype(dtype)
      yield
      torch.set_default_dtype(torch.float)

    # Use the correct variant string to get the matching model configuration.
    model_config = get_model_config(VARIANT)
    model_config.tokenizer = os.path.join(weights_dir, "tokenizer.model")
    model_config.dtype = "bfloat16" if MACHINE_TYPE == "cuda" else "float32"

    # --- Model Instantiation and Loading ---
    device = torch.device(MACHINE_TYPE)
    with _set_default_tensor_type(model_config.get_dtype()):
      model = GemmaForCausalLM(model_config)
      model.load_weights(weights_file)
      model = model.to(device)

    # Move the 'freqs_cis' buffer to the correct device to prevent runtime errors.
    if MACHINE_TYPE == 'cuda':
        model.freqs_cis = model.freqs_cis.to(device)
    
    model = model.eval()
    print("Gemma 2 model loaded and patched successfully.")

except Exception as e:
    print(f"ERROR: Failed to load Gemma 2 model. {e}")



#test

USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn><eos>\n"

prompt = (
    USER_CHAT_TEMPLATE.format(
        prompt="What is a good place for travel in the US?"
    )
    + MODEL_CHAT_TEMPLATE.format(prompt="California.")
    + USER_CHAT_TEMPLATE.format(prompt="What can I do in California?")
    + "<start_of_turn>model\n"
)

model.generate(
    USER_CHAT_TEMPLATE.format(prompt=prompt),
    device=device,
    output_len=100,
)


from gemma.tokenizer import Tokenizer


class ImprovedLLMSolver:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
        self.MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"
        # FIX: Instantiate the tokenizer to use for the safety check.
        self.tokenizer = Tokenizer(model.config.tokenizer)

    def format_prompt(self, task):
        system_instruction = (
            "You are an expert ARC puzzle solver. Analyze the training examples to find the rule. "
            "First, state the rule in one sentence. Then, on a new line, provide ONLY the output grid as a Python list."
        )
        context_prompt = ""
        for i, pair in enumerate(task['train']):
            # FIX: Compress the grid representation by removing whitespace to save tokens.
            input_grid_str = str(pair['input']).replace(' ', '')
            output_grid_str = str(pair['output']).replace(' ', '')
            context_prompt += f"--- Training Example {i+1} ---\nInput: {input_grid_str}\nOutput: {output_grid_str}\n\n"
        
        test_input_grid_str = str(task['test'][0]['input']).replace(' ', '')
        context_prompt += f"--- Test Case ---\nInput: {test_input_grid_str}\n"
        
        full_prompt = (
            self.USER_CHAT_TEMPLATE.format(prompt=system_instruction) +
            self.MODEL_CHAT_TEMPLATE.format(prompt="Understood. I will state the rule, then provide the grid.") +
            self.USER_CHAT_TEMPLATE.format(prompt=context_prompt) +
            "<start_of_turn>model\n"
        )
        return full_prompt

    def parse_llm_response(self, response_text):
        match = re.search(r'\[\s*\[.*\]\s*\]', response_text, re.DOTALL)
        if match:
            grid_str = match.group(0)
            try:
                predicted_grid = ast.literal_eval(grid_str)
                if isinstance(predicted_grid, list) and all(isinstance(row, list) for row in predicted_grid):
                    return predicted_grid
            except (ValueError, SyntaxError): pass
        return [[0]]

    def solve(self, task):
        if not self.model or not task.get("test"):
            return [[[0]]] if task.get("test") else []
        
        prompt = self.format_prompt(task)
        output_len = 300

        # --- FIX: SAFETY CHECK FOR PROMPT LENGTH ---
        prompt_tokens = self.tokenizer.encode(prompt)
        max_len = self.model.config.max_position_embeddings
        if len(prompt_tokens) + output_len >= max_len:
            print(f"WARNING: Prompt for task is too long ({len(prompt_tokens)} tokens) and would cause a crash. Skipping LLM attempt.")
            # Return the default fallback grid in the expected list-of-lists format
            return [[[0]]]
            
        response_text = self.model.generate(prompt, device=self.device, output_len=output_len)
        generated_part = response_text[len(prompt):]
        
        # print("-" * 50 + "\nDEBUG: Raw LLM Output:\n" + generated_part + "\n" + "-" * 50)
        
        predicted_grid = self.parse_llm_response(generated_part)
        return [predicted_grid]




# --- Test the Improved LLM Solver with Gemma 2 ---
if DEVICE == "cuda" and model is not None:
    sample_task_id = '00576224'
    llm_solver = ImprovedLLMSolver(model, DEVICE)
    sample_task = train_challenges.get(sample_task_id)

    if sample_task:
        print(f"\nTesting Improved LLM Solver (Gemma 2) on task: {sample_task_id}")
        llm_predictions = llm_solver.solve(sample_task)
        
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs[0].imshow(sample_task['test'][0]['input'], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[0].set_title("Test Input")
        axs[1].imshow(llm_predictions[0], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[1].set_title("LLM Prediction")
        axs[2].imshow(train_solutions[sample_task_id][0], cmap=ARC_COLORMAP, norm=ARC_NORM)
        axs[2].set_title("Ground Truth")
        for ax in axs: ax.axis('off')
        plt.show()
else:
    print("\nSkipping LLM solver test as no GPU is detected or model failed to load.")


def exact_match(pred_grid, true_grid):
    """Checks if two grids are identical."""
    return np.array(pred_grid).tolist() == np.array(true_grid).tolist()


def evaluate_solver(challenges, solutions, heuristic_solver, llm_solver):
    """
    Evaluates the hybrid solver on a given dataset (e.g., the evaluation set).
    """
    total_tasks = 0
    correct_heuristic = 0
    correct_llm = 0
    correct_hybrid = 0
    
    # We'll use a smaller subset for this example to run faster
    # On submission, you'd run on the full evaluation set.
    task_ids = list(challenges.keys())[:20]
    for task_id in tqdm(task_ids, desc="Evaluating Hybrid Solver"):
        task_data = challenges[task_id]
        true_solutions = solutions[task_id]
        
        # We assume one test case per task for this evaluation logic
        if len(task_data['test']) != 1: continue
        total_tasks += 1

        # Attempt 1: Heuristic
        heuristic_pred = heuristic_solver.solve(task_data)[0]
        is_heuristic_correct = exact_match(heuristic_pred, true_solutions[0])
        if is_heuristic_correct:
            correct_heuristic += 1
        
        # Attempt 2: LLM (only run if GPU is available)
        is_llm_correct = False
        if DEVICE == 'cuda':
            llm_pred = llm_solver.solve(task_data)[0]
            is_llm_correct = exact_match(llm_pred, true_solutions[0])
            if is_llm_correct:
                correct_llm += 1
        
        if is_heuristic_correct or is_llm_correct:
            correct_hybrid += 1
    print("\n--- Evaluation Results ---")
    if total_tasks > 0:
        heuristic_acc = (correct_heuristic / total_tasks) * 100
        llm_acc = (correct_llm / total_tasks) * 100
        hybrid_acc = (correct_hybrid / total_tasks) * 100
        
        print(f"Total tasks evaluated: {total_tasks}")
        print(f"Heuristic Solver Accuracy: {heuristic_acc:.2f}%")
        print(f"LLM Solver Accuracy: {llm_acc:.2f}%")
        print(f"Hybrid (Either Correct) Accuracy: {hybrid_acc:.2f}%")
    else:
        print("No tasks were evaluated.")


# Run the evaluation with the new solver
import re 

if eval_challenges and eval_solutions:
    llm_solver_instance = ImprovedLLMSolver(model, DEVICE) if model else None
    evaluate_solver(eval_challenges, eval_solutions, HeuristicSolver(), llm_solver_instance)
else:
    print("Skipping evaluation as evaluation data is not available.")


def generate_submission(challenges, heuristic_solver, llm_solver):
    submission_dict = {}
    for task_id, task in tqdm(challenges.items(), desc="Generating Submission"):
        heuristic_preds = heuristic_solver.solve(task)
        
        llm_preds = []
        if DEVICE == 'cuda' and llm_solver and llm_solver.model:
            if task.get('test'):
                llm_pred_first = llm_solver.solve(task)[0]
                llm_preds.append(llm_pred_first)
                if len(task['test']) > 1: llm_preds.extend(heuristic_preds[1:])
        else:
            llm_preds = heuristic_preds

        task_predictions = []
        for i in range(len(task.get('test', []))):
            task_predictions.append({ "attempt_1": heuristic_preds[i], "attempt_2": llm_preds[i] })
        submission_dict[task_id] = task_predictions
    return submission_dict



final_heuristic_solver = HeuristicSolver()

final_llm_solver = ImprovedLLMSolver(model, DEVICE) if model else None


submission = generate_submission(test_challenges, final_heuristic_solver, final_llm_solver)


submission_path = "submission.json"
with open(submission_path, "w") as f:
    json.dump(submission, f)


print(f"\n✅ Submission file saved to {submission_path}")


print("\n--- Sample Submission Preview ---")
sample_ids_in_submission = list(submission.keys())[:2]
for task_id in sample_ids_in_submission:
    print(f"\nTask ID: {task_id}")
    for i, pred_pair in enumerate(submission[task_id]):
        print(f"  Test Case #{i+1}:")
        print(f"    Attempt 1: {str(pred_pair['attempt_1'])[:80]}...")
        print(f"    Attempt 2: {str(pred_pair['attempt_2'])[:80]}...")

