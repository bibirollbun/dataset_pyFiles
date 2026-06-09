!pip install -q --no-index --find-links=/kaggle/input/vllm-lib vllm


import os
os.environ['VLLM_USE_V1'] = '0'
debug = not os.getenv('KAGGLE_IS_COMPETITION_RERUN')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
import re
import time
import random
import warnings
from collections import Counter
import numpy as np, pandas as pd, polars as pl

import torch
import vllm
from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')
print('PyTorch version:', torch.__version__)
print('vLLM:', vllm.__version__)


class ARCvLLMPredictor:
    def __init__(self, model_path, tensor_parallel_size=1, trust_remote_code=True):
        self.llm = None
        self.sampling_params = None
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                max_model_len=32768, 
                gpu_memory_utilization=0.965,
                max_num_seqs=32
            )
            print(f"INFO: vLLM would be initialized with model: {model_path}")
            print("INFO: Ensure vLLM is installed and the above lines are uncommented.")
        except ImportError:
            print("ERROR: vLLM library not found. Please install vLLM to use this predictor.")
            raise
        except Exception as e:
            print(f"ERROR: Could not initialize vLLM with model {model_path}: {e}")
            raise

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95, 
            max_tokens=8192,
        )

    def predict(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params,chat_template_kwargs={"enable_thinking": True})
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

arc_predictor = ARCvLLMPredictor("/kaggle/input/qwen-3/transformers/8b-awq/1", tensor_parallel_size=4)


def load_arc_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def get_task_data(tasks, task_id):
    if tasks and task_id in tasks: return tasks[task_id]
    print(f"Error: Task ID '{task_id}' not found in challenges.")
    return None

def format_grid_to_text(grid_array):
    if not isinstance(grid_array, list) or \
       not all(isinstance(row, list) for row in grid_array) or \
       not all(isinstance(val, int) for row in grid_array for val in row):
        return "[[0]]"
    return str(grid_array).replace(" ", "")



import os
import json

if debug:
    print(debug)
    split = "evaluation"
    with open(f'../input/arc-prize-2025/arc-agi_{split}_challenges.json', 'r') as f:
        all_challenges = json.load(f)
else:
    print(debug)
    split = "test"
    with open(f'../input/arc-prize-2025/arc-agi_{split}_challenges.json', 'r') as f:
        all_challenges = json.load(f)



def create_arc_prompts_for_task(task_data_instance, task_id=""):
    prompts = []
    for test_idx in range(len(task_data_instance['test'])):
        prompt = f"Solve the ARC task '{task_id}'.\n"
        prompt += "The grids use integers 0-9 for colors.\n"
        prompt += "Analyze the 'train' examples to understand the transformation rule, then apply it to the 'test' input.\n\n"

        prompt += "Train Examples:\n"
        for i, pair in enumerate(task_data_instance['train']):
            prompt += f"  Train Pair {i+1}:\n"
            prompt += f"    Input: {format_grid_to_text(pair['input'])}\n"
            prompt += f"    Output: {format_grid_to_text(pair['output'])}\n"

        prompt += f"\nTest Input #{test_idx+1}:\n"
        prompt += f"  Input: {format_grid_to_text(task_data_instance['test'][test_idx]['input'])}\n\n"
        prompt += "Based on the train examples, what is the corresponding output grid for this test input?\n"
        prompt += "Provide ONLY the output grid as a Python list of lists (e.g., [[1,2],[3,4]]).\n"
        prompt += "Output Grid:"

        prompts.append(prompt)
    return prompts


import numpy as np
import os
import re

def parse_llm_response_to_grid(llm_output_text):
    llm_output_text = llm_output_text.strip()

    grid_pattern = r'(\[\[\s*(?:\d\s*,\s*)*\d?\s*\](?:\s*,\s*\[\s*(?:\d\s*,\s*)*\d?\s*\])*\s*\])'

    match = re.search(grid_pattern, llm_output_text)

    if match:
        grid_str = match.group(1)

        try:
            predicted_grid = json.loads(grid_str)
            if not isinstance(predicted_grid, list):
                print(f"Warning: Parsed structure is not a list: {predicted_grid}")
                return [[0]]
            if not predicted_grid:
                return [[0]]

            if not all(isinstance(row, list) for row in predicted_grid):
                print(f"Warning: Parsed structure is not a list of lists: {predicted_grid}")
                return [[0]]

            for row_idx, row in enumerate(predicted_grid):
                if not all(isinstance(val, int) and 0 <= val <= 9 for val in row):
                    print(f"Warning: Row {row_idx} contains non-ARC values (not int 0-9): {row}")
                    return [[0]]
            if predicted_grid and predicted_grid[0]:
                first_row_len = len(predicted_grid[0])
                if not all(len(row) == first_row_len for row in predicted_grid):
                    print(f"Warning: Non-rectangular grid from LLM after parsing: {predicted_grid}")
                    return [[0]]
            elif predicted_grid and not any(predicted_grid):
                 pass 

            return predicted_grid

        except json.JSONDecodeError as e:
            print(f"Error: JSON decoding failed for extracted grid string: '{grid_str}'. Error: {e}")
            return [[0]]
        except Exception as e:
            print(f"Error: Unexpected error processing extracted grid string: '{grid_str}'. Error: {e}")
            return [[0]]

    print(f"Warning: Could not find a valid grid pattern in LLM output (first 200 chars): '{llm_output_text[:200]}...'")
    return [[0]]


all_prompts_attempt1 = []
all_prompts_attempt2 = []

prompt_metadata = []

print("\nGenerating all prompts...")
for task_id, task_data in all_challenges.items():
    if task_data:
        prompts_for_task_a1 = create_arc_prompts_for_task(task_data, task_id)
        for i, p_a1 in enumerate(prompts_for_task_a1):
            all_prompts_attempt1.append(p_a1)
            prompt_metadata.append({"task_id": task_id, "test_idx": i, "attempt": 1, "original_prompt_index": len(all_prompts_attempt1)-1})

            p_a2 = p_a1 + "\nTry a different approach. Output Grid:"
            all_prompts_attempt2.append(p_a2)

print(f"Total prompts for attempt 1: {len(all_prompts_attempt1)}")
print(f"Total prompts for attempt 2: {len(all_prompts_attempt2)}")

all_predictions_map = {}


def compare_grids(grid1, grid2):

    if grid1 is None or grid2 is None:
        return False
    if not isinstance(grid1, list) or not isinstance(grid2, list):
        return False
    if len(grid1) == 0 and len(grid2) == 0:
        return True
    if len(grid1) == 0 or len(grid2) == 0:
        return False
    if all(not row for row in grid1) and all(not row for row in grid2):
        return len(grid1) == len(grid2)
    if all(not row for row in grid1) or all(not row for row in grid2):
        return False

    if len(grid1) != len(grid2):
        return False

    for i in range(len(grid1)):
        row1 = grid1[i]
        row2 = grid2[i]
        if not isinstance(row1, list) or not isinstance(row2, list):
            return False
        if len(row1) != len(row2):
            return False
        for j in range(len(row1)):
            if row1[j] != row2[j]:
                return False
    return True

def evaluate_submission(predictions_file, solutions_file):
    try:
        with open(predictions_file, 'r') as f:
            predictions_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Predictions file '{predictions_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from predictions file '{predictions_file}'.")
        return

    try:
        with open(solutions_file, 'r') as f:
            solutions_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Solutions file '{solutions_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from solutions file '{solutions_file}'.")
        return

    total_tasks_solved = 0
    total_tasks_evaluated = 0
    details = []

    for task_id, predicted_task_cases in predictions_map.items():
        total_tasks_evaluated += 1
        task_detail = {"task_id": task_id, "status": "Not Solved", "test_cases": []}

        if task_id not in solutions_map:
            task_detail["status"] = "Solution Not Found"
            details.append(task_detail)
            print(f"Warning: Solutions for task '{task_id}' not found in solutions file. Skipping.")
            continue

        solution_task_cases = solutions_map[task_id]

        if not isinstance(predicted_task_cases, list) or not isinstance(solution_task_cases, list):
            task_detail["status"] = "Malformed prediction or solution data"
            details.append(task_detail)
            print(f"Warning: Prediction or solution data for task '{task_id}' is malformed. Skipping.")
            continue
            
        if len(predicted_task_cases) != len(solution_task_cases):
            task_detail["status"] = f"Mismatch in number of test cases (Pred: {len(predicted_task_cases)}, Sol: {len(solution_task_cases)})"
            print(f"Warning: Mismatch in number of test cases for task '{task_id}'. Predicted: {len(predicted_task_cases)}, Solution: {len(solution_task_cases)}. This task cannot be 'Solved'.")
        
        all_test_cases_for_this_task_solved = True
        if not predicted_task_cases:
             all_test_cases_for_this_task_solved = False


        num_test_cases_to_compare = min(len(predicted_task_cases), len(solution_task_cases))

        for test_idx in range(num_test_cases_to_compare):
            prediction_info = predicted_task_cases[test_idx] # Dict with "attempt_1", "attempt_2"
            
            if not solution_task_cases[test_idx]:
                print(f"Warning: No solution attempts found for task '{task_id}', test case {test_idx}. Skipping this test case.")
                all_test_cases_for_this_task_solved = False # Cannot be solved if solution is missing
                task_detail["test_cases"].append({"test_idx": test_idx, "status": "Solution Missing"})
                continue
                
            actual_solution_grid = solution_task_cases[test_idx][0] # The first solution grid

            predicted_grid_a1 = prediction_info.get("attempt_1")
            predicted_grid_a2 = prediction_info.get("attempt_2")

            solved_by_a1 = compare_grids(predicted_grid_a1, actual_solution_grid)
            solved_by_a2 = compare_grids(predicted_grid_a2, actual_solution_grid)
            
            current_test_case_solved = solved_by_a1 or solved_by_a2
            test_case_status = "Correct" if current_test_case_solved else "Incorrect"
            if solved_by_a1 and solved_by_a2:
                test_case_status += " (Both attempts)"
            elif solved_by_a1:
                test_case_status += " (Attempt 1)"
            elif solved_by_a2:
                test_case_status += " (Attempt 2)"


            task_detail["test_cases"].append({"test_idx": test_idx, "status": test_case_status})

            if not current_test_case_solved:
                all_test_cases_for_this_task_solved = False
        
        if len(predicted_task_cases) != len(solution_task_cases):
            all_test_cases_for_this_task_solved = False


        if all_test_cases_for_this_task_solved and num_test_cases_to_compare > 0 :
            total_tasks_solved += 1
            task_detail["status"] = "Solved"
        
        details.append(task_detail)

    print("\n--- Evaluation Summary ---")
    print(f"Total Tasks Evaluated: {total_tasks_evaluated}")
    print(f"Total Tasks Solved:    {total_tasks_solved}")
    if total_tasks_evaluated > 0:
        accuracy = (total_tasks_solved / total_tasks_evaluated) * 100
        print(f"Overall Accuracy:      {accuracy:.4f}%")
    else:
        print("No tasks were evaluated.")
    
    return total_tasks_solved, total_tasks_evaluated, details


if arc_predictor and all_prompts_attempt1:
    print("\nProcessing all prompts for Attempt 1...")
    generated_texts_a1 = arc_predictor.predict(all_prompts_attempt1)
    
    print("\nProcessing all prompts for Attempt 2...")
    generated_texts_a2 = arc_predictor.predict(all_prompts_attempt2)

    print("\nParsing results and organizing predictions...")

    current_meta_idx = 0
    for meta in prompt_metadata:
        if meta["attempt"] == 1:
            task_id = meta["task_id"]
            test_idx = meta["test_idx"]
            original_prompt_idx = meta["original_prompt_index"]

            if task_id not in all_predictions_map:
                num_test_cases_for_task = len(all_challenges[task_id].get('test', []))
                all_predictions_map[task_id] = [{} for _ in range(num_test_cases_for_task)]


            parsed_grid_a1 = parse_llm_response_to_grid(generated_texts_a1[original_prompt_idx])
            parsed_grid_a2 = parse_llm_response_to_grid(generated_texts_a2[original_prompt_idx])

            if test_idx < len(all_predictions_map[task_id]):
                all_predictions_map[task_id][test_idx] = {
                    "attempt_1": parsed_grid_a1,
                    "attempt_2": parsed_grid_a2
                }
            else:
                print(f"Warning: test_idx {test_idx} out of bounds for task {task_id} while assigning predictions.")

output_submission_file = 'submission.json'

print(f"\nSaving all predictions to: {output_submission_file}")
with open(output_submission_file, 'w') as f:
    json.dump(all_predictions_map, f, indent=4) 


if debug:
    evaluate_submission(output_submission_file, '/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json')

