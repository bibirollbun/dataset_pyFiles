import os
import sys

# Platform detection and configuration
platform_env = (
    "kaggle"
    if "KAGGLE_CONTAINER_NAME" in os.environ
    else "colab"
    if "COLAB_JUPYTER_IP" in os.environ
    else "other"
)

if platform_env == "kaggle":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    llm_model_pth = (
        "/kaggle/input/casperhansen-deepseek-r1-distill-qwen-14b-awq/transformers/default/1"
    )
    eval_file = "/kaggle/input/aime-2025/aime_2025.csv"
    # make sure to pin the completely loaded model version, because once notebook gets saved dataset (synced with notebook) will
    # point to version with empty torchcompile version
    torch_compile_cache_folder = "/kaggle/input/0325-vllm-torch-cache/torch_compile_cache"
elif platform_env == "colab":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm_model_pth = "/content/kaggle/input/casperhansen/deepseek-r1-distill-qwen-1.5b-awq"
    eval_file = "kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    sys.path.append("kaggle/input/ai-mathematical-olympiad-progress-prize-2")
    # this path should be added before import kaggle_evaluation.aimo_2_inference_server
    # os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    torch_compile_cache_folder = None


import os
import sys
import platform
import json
import shutil
import asyncio
import re
import time
import random
import math
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import polars as pl
from pprint import pprint
from datetime import datetime
import pytz
from tqdm.auto import tqdm
import pathlib
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from pathlib import Path
import logging

import kaggle_evaluation.aimo_2_inference_server

import torch

use_uvloop = True
if use_uvloop:
    import uvloop

    uvloop.install()
    print("Using uvloop as the asyncio event loop policy")


warnings.simplefilter("ignore")


# Environment variable configurations
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
# https://github.com/vllm-project/vllm/pull/11394
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
# spawn takes a lot of time, safer but slower and also many of the logs not shown
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
# https://github.com/vllm-project/vllm/pull/14138
os.environ["VLLM_MARLIN_USE_ATOMIC_ADD"] = "0"

os.environ["VLLM_USE_V1"] = "0"

if os.environ["VLLM_USE_V1"] == "1":
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN_VLLM_V1"
else:
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"

# for reproducibility
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


print("PyTorch version:", torch.__version__)

# Generation hyperparameters
# VARY_SAMPLING_PARAMS = True
VARY_TEMPERATURE = True
VARY_TOP_P = True
MAX_NUM_SEQS = 16
MAX_MODEL_LEN = int(8192 * 2)
MAX_TIME_PER_PROBLEM = int(
    4.5 * 60
)  # 50 probs * 5 = 250 mins + 10 mins model loading + 10-20 minutes for longer solutions with timeout extension
TIMEOUT_EXTENSION = (
    210  # if too high very difficult problems may not return any answer and still time is lost
)
# Define the minimum percentage of tasks that should be completed before considering an extension
PRE_TIMEOUT_COMPLETION_PERC = 0.35  # pushes towards extended timeout if higher
# Define the percentage threshold for early stopping based on answer consensus
EARLY_STOP_PERC = 0.35  # If >= 35% of sequences produce the same valid integer answer, stop early
LOGPROB_WEIGHTING = True


def seed_everything(seed):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


SEED = 2024
seed_everything(seed=SEED)

start_time = time.time()
cutoff_time = start_time + (4 * 60 + 46) * 60
cutoff_times = [int(x) for x in np.linspace(cutoff_time, start_time + 60 * 60, 50 + 1)]




class GenerationLogger:
    """
    Tracks all problem generation attempts and metrics in a single comprehensive CSV file.
    Minimizes overhead during competition runs.
    """

    def __init__(self, minimal_logging=False):
        """
        Initialize the generation logger with global parameters.

        Args:
            minimal_logging: Whether to use minimal logging (True during competition)
        """
        # Create timestamp for the CSV filename (IST time)
        ist = pytz.timezone("Asia/Kolkata")
        timestamp = datetime.now(ist).strftime("%d%H%M")

        # Create output directory if it doesn't exist
        os.makedirs("problem_outputs", exist_ok=True)

        # CSV path for comprehensive tracking
        self.csv_path = f"problem_outputs/{timestamp}.csv"
        self.minimal_logging = minimal_logging
        print(f"{minimal_logging=}")
        self.rows = []

        # Get global parameters from environment
        eval_file_path = Path(eval_file)
        self.global_info = {
            # Global hyperparameters
            # "VARY_SAMPLING_PARAMS": VARY_SAMPLING_PARAMS,
            "VARY_TEMPERATURE": VARY_TEMPERATURE,
            "VARY_TOP_P": VARY_TOP_P,
            "MAX_NUM_SEQS": MAX_NUM_SEQS,
            "MAX_MODEL_LEN": MAX_MODEL_LEN,
            "MAX_TIME_PER_PROBLEM": MAX_TIME_PER_PROBLEM,
            "TIMEOUT_EXTENSION": TIMEOUT_EXTENSION,
            "PRE_TIMEOUT_COMPLETION_PERC": PRE_TIMEOUT_COMPLETION_PERC,
            "EARLY_STOP_PERC": EARLY_STOP_PERC,
            "LOGPROB_WEIGHTING": LOGPROB_WEIGHTING,
            # Model info
            "llm_model_pth": llm_model_pth,
            "eval_file": eval_file_path.stem,  # Just the filename
            # vLLM environment variables
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", ""),
            "VLLM_USE_V1": os.environ.get("VLLM_USE_V1", ""),
            "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD", ""),
            "VLLM_MARLIN_USE_ATOMIC_ADD": os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", ""),
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND", ""),
        }

        # Initialize CSV if it doesn't exist yet
        if not os.path.exists(self.csv_path) and not self.minimal_logging:
            # Create empty DataFrame with column names
            pd.DataFrame(columns=self._get_column_names()).to_csv(self.csv_path, index=False)

    def _get_column_names(self) -> List[str]:
        """Define all columns for the comprehensive CSV."""
        # Basic problem info
        basic_cols = ["problem_id", "question"]

        # Generation info
        generation_cols = [
            "request_id",
            "temperature",
            "top_p",
            "num_tokens",
            "time_taken",
            "avg_logprob",
            "stop_reason",  # Can now be 'early_stopped'
            "timeout_type",
            "extracted_answer",
            "is_complete",
        ]

        # Generation text (can be large)
        text_cols = ["generated_text"]

        # Global settings (same for all rows but useful for filtering)
        global_cols = list(self.global_info.keys())

        return basic_cols + generation_cols + text_cols + global_cols

    def add_generation_result(
        self,
        problem_id: str,
        question: str,
        request_id: str,
        temperature: float,
        top_p: float,
        num_tokens: int,
        time_taken: float,
        generated_text: str,
        stop_reason: str,
        timeout_type: str,
        avg_logprob: Optional[float] = None,
        extracted_answer: Optional[str] = None,
        is_complete: bool = True,
    ):
        """
        Add a single generation result to the tracker.

        Args:
            problem_id: Unique identifier for the problem
            question: The question text
            request_id: Unique identifier for this generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            num_tokens: Number of tokens generated
            time_taken: Time taken for generation in seconds
            generated_text: The generated text
            stop_reason: Reason for stopping (e.g., 'finished', 'max_length', 'timeout', 'early_stopped')
            timeout_type: Type of timeout ('primary' or 'extended')
            extracted_answer: The extracted answer from boxed text if available
            is_complete: Whether generation completed successfully (False for timeout or early_stopped cancellations)
        """

        # Skip all processing if in minimal logging mode
        if self.minimal_logging:
            return

        row_data = {
            # Basic cols (from _get_column_names)
            "problem_id": str(problem_id),
            "question": question[:200] + "..." if len(question) > 200 else question,
            # Generation cols (from _get_column_names)
            "request_id": request_id,
            "temperature": temperature,
            "top_p": top_p,
            "num_tokens": num_tokens,
            "time_taken": time_taken,
            "avg_logprob": avg_logprob
            if avg_logprob is not None
            else float("nan"),  # Log avg_logprob
            "stop_reason": stop_reason,
            "timeout_type": timeout_type,
            "extracted_answer": extracted_answer,
            "is_complete": is_complete,
            # Text cols (from _get_column_names)
            "generated_text": generated_text,
        }

        # Add global info to each row
        row_data.update(self.global_info)

        # Add to rows list
        self.rows.append(row_data)

    def save_to_csv(self):
        """Save all tracked generations to CSV."""
        if self.minimal_logging or not self.rows:
            return

        try:
            # Convert to DataFrame and append to CSV
            df = pd.DataFrame(self.rows)

            if os.path.exists(self.csv_path):
                # Append without header
                df.to_csv(self.csv_path, mode="a", header=False, index=False)
            else:
                # Create new file with header
                df.to_csv(self.csv_path, index=False)

            # Clear rows after saving
            self.rows = []
            print(f"Saved {len(df)} generation results to {self.csv_path}")

        except Exception as e:
            print(f"Error saving to CSV: {e}")


# Create the generation logger - use minimal logging during competition
logger = GenerationLogger(minimal_logging=os.getenv("KAGGLE_IS_COMPETITION_RERUN") is not None)





def extract_boxed_text(text):
    """Extract answer from boxed text in the generated output."""
    pattern = r"oxed{(.*?)}"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""



def batch_message_filter(
    list_of_messages: List[Tuple[List[Dict], float]],
) -> Tuple[List[List[Dict]], List[Tuple[str, float]]]:
    """Filter messages to extract answers and their associated avg_logprob."""
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages, avg_logprob in list_of_messages:
        answer = extract_boxed_text(messages[-1]["content"])
        if answer:
            extracted_answers.append((answer, avg_logprob))
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers



def select_answer(answers: List[Tuple[str, float]]):
    """
    Select the final answer from multiple candidates using voting.
    Optionally weights votes by avg_logprob if LOGPROB_WEIGHTING is True.
    """
    print(f"\nInput answers with avg_logprob for voting: {answers}")
    print(f"Logprob weighting enabled: {LOGPROB_WEIGHTING}")
    counter = Counter()
    processed_count = 0
    answer_weights = {}

    for answer_candidate, avg_logprob in answers:
        try:
            numeric_val = pd.to_numeric(answer_candidate, errors="coerce")

            if not pd.isna(numeric_val) and numeric_val == int(numeric_val):
                num_int = int(numeric_val)
                key = num_int % 1000

                weight = float("nan")
                if LOGPROB_WEIGHTING:
                    if avg_logprob is not None and not math.isnan(avg_logprob):
                        # Use exp(avg_logprob). Higher avg_logprob -> higher weight (closer to 1).
                        # avg_logprob is typically negative.
                        weight = math.exp(avg_logprob)
                        print(
                            f"  Weighting answer {key} with avg_logprob {avg_logprob:.4f}: weight = exp({avg_logprob:.4f}) = {weight:.4f}"
                        )
                    else:
                        # Assign a default neutral weight if avg_logprob is invalid
                        weight = 1.0  # Or perhaps a lower default like exp(-5)? Using 1.0 for now.
                        print(
                            f"  Answer {key} has invalid avg_logprob ({avg_logprob}), using default weight: {weight:.4f}"
                        )
                else:
                    # Default random weighting for tie-breaking
                    weight = 1.0 + random.random() / 1_000
                    print(f"  Weighting answer {key} (no logprob weighting): weight = {weight:.4f}")

                counter[key] += weight
                answer_weights[key] = answer_weights.get(key, []) + [weight]
                processed_count += 1
            else:
                print(
                    f"Skipping non-whole number or NaN value: {answer_candidate!r} (became {numeric_val!r})"
                )
        except Exception as e:
            # This will catch ANY standard exception that occurred anywhere in the 'try' block
            # for the current answer_candidate (e.g., TypeError from math.isnan, ValueError from int(), etc.)
            print(f"Error processing candidate {answer_candidate!r} for voting: {e}")

    print(f"Processed {processed_count} valid whole numbers for voting.")
    print(f"Weighted Voting Counter: {counter}")
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        print(f"Weights per answer key: {answer_weights}")

    if not counter:
        print("Voting counter is empty, returning default 210.")
        return 210

    # Find the key (answer) with the highest value (count)
    sorted_items = sorted([(v, k) for k, v in counter.items()], reverse=True)
    print(f"Sorted voting items (count, answer): {sorted_items}")
    _, most_common_answer = sorted_items[0]

    print(f"Selected answer via voting: {most_common_answer}")
    return most_common_answer


async def batch_message_generate(
    original_list_of_messages, problem_id: str, question: str
) -> Tuple[List[List[Dict]], Optional[int]]:
    """
    Generates responses for a batch of message lists, logging results and implementing early stopping.

    Args:
        original_list_of_messages: The initial list of message histories.
        problem_id: The unique ID for the problem.
        question: The original question string.

    Returns:
        A tuple containing:
        - A list of completed message histories (original + assistant response).
        - An integer answer if early stopping was triggered, otherwise None.
    """
    max_tokens = MAX_MODEL_LEN
    start_time_batch = time.time()
    current_max_time = MAX_TIME_PER_PROBLEM
    extended_once = False
    timeout_type = "primary"
    early_stop_triggered = False
    early_stopped_answer = None
    early_stop_answer_counter = Counter()  # For tracking answers during generation
    early_stop_threshold = math.ceil(EARLY_STOP_PERC * len(original_list_of_messages))
    print(f"Early stopping threshold: {early_stop_threshold} identical valid answers.")

    # Define parameter ranges based on the granular flags
    temp_choices = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0] if VARY_TEMPERATURE else [1.0]
    top_p_choices = [0.9, 0.95] if VARY_TOP_P else [1.0]
    param_combinations = sorted(
        [(temp, top_p) for temp in temp_choices for top_p in top_p_choices],
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )

    # Assign parameters cyclically
    assigned_params = [
        param_combinations[i % len(param_combinations)]
        for i in range(len(original_list_of_messages))
    ]

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )
        for messages in original_list_of_messages
    ]

    # Dictionary to store the last known output for potentially incomplete tasks
    partial_outputs_dict = {}
    # Map task object to its original index
    task_to_index_map = {}

    # Modified coroutine to update partial_outputs_dict and log results
    async def process_single_generation(text, index):
        gen_start_time = time.time()
        task_temperature, task_top_p = assigned_params[index]

        task_sampling_params = SamplingParams(
            temperature=task_temperature,
            top_p=task_top_p,
            skip_special_tokens=True,
            max_tokens=max_tokens,
            stop=["</think>"],
            seed=SEED,
            min_p=0.05,
            logprobs=5,
            # repetition_penalty=1.1,
        )

        param_info = {"temp": task_temperature, "top_p": task_top_p}

        # Create a request_id that includes parameter info
        request_id = f"{random.random()}_{param_info['temp']}_{param_info['top_p']}"

        outputs = []
        last_output = None
        avg_logprob = float("nan")  # Initialize avg_logprob

        try:
            async for output in llm.generate(
                prompt=text,
                sampling_params=task_sampling_params,
                request_id=request_id,
            ):
                outputs.append(output)
                last_output = output
                # Update the shared dictionary with the latest output
                partial_outputs_dict[index] = last_output

            # Calculate avg_logprob if generation finished
            if last_output and last_output.finished and last_output.outputs:
                completion_output = last_output.outputs[0]
                cum_logprob = completion_output.cumulative_logprob
                num_gen_tokens = len(completion_output.token_ids)
                if num_gen_tokens > 0 and cum_logprob is not None:
                    avg_logprob = cum_logprob / num_gen_tokens
                    print(
                        f"  Index {index}: Avg Logprob = {avg_logprob:.4f} (Cumulative: {cum_logprob:.4f}, Tokens: {num_gen_tokens})"
                    )
                else:
                    # print(f"  Index {index}: Cannot calculate avg_logprob (Tokens: {num_gen_tokens}, Cumulative: {cum_logprob})")
                    avg_logprob = float("nan")  # Mark as NaN
            else:
                # print(f"  Index {index}: Generation did not finish or no outputs, cannot calculate avg_logprob.")
                avg_logprob = float("nan")  # Mark as NaN

        except Exception as e:
            print(f"Exception during generation for index {index}: {e}")
            avg_logprob = float("nan")  # Mark as NaN on error

        gen_time_taken = time.time() - gen_start_time
        return outputs, index, param_info, gen_time_taken, request_id, avg_logprob

    # Start all tasks concurrently
    tasks = []
    for i, text in enumerate(list_of_texts):
        task = asyncio.create_task(process_single_generation(text, i))
        tasks.append(task)
        task_to_index_map[task] = i  # Store mapping

    total_tasks_count = len(tasks)  # Store the initial count

    # Set up progress bar
    pbar = tqdm(total=total_tasks_count, desc="Generating responses")

    # Use asyncio.wait with a timeout
    done_tasks = set()
    pending_tasks = set(tasks)
    task_results = {}  # Stores (list_of_outputs, params_info, time_taken, request_id) for completed tasks
    task_params_used = {i: assigned_params[i] for i in range(total_tasks_count)}
    tasks_processed_count = 0
    last_status_count = 0

    # Main execution loop with timeout and early stopping check
    while pending_tasks and not early_stop_triggered:  # Added early stop check
        done, pending = await asyncio.wait(
            pending_tasks,
            timeout=5,
            return_when=asyncio.FIRST_COMPLETED,
        )

        newly_completed = 0
        for task in done:
            index = task_to_index_map.get(task, -1)  # Get index for this task
            if index == -1:
                continue  # Should not happen

            try:
                outputs_list, _, task_params_info, time_taken, req_id, avg_logprob = task.result()
                task_results[index] = (
                    outputs_list,
                    task_params_info,
                    time_taken,
                    req_id,
                    avg_logprob,
                )
                done_tasks.add(task)
                newly_completed += 1

                # --- Early Stopping Check ---
                if outputs_list:
                    final_output = outputs_list[-1]
                    generated_text = final_output.outputs[0].text
                    boxed_text = extract_boxed_text(generated_text)
                    if boxed_text:  # Only check if an answer was extracted
                        try:
                            numeric_val = pd.to_numeric(boxed_text, errors="coerce")
                            if not pd.isna(numeric_val) and numeric_val == int(numeric_val):
                                num_int = int(numeric_val)
                                key = num_int % 1000
                                early_stop_answer_counter[key] += 1
                                print(
                                    f"  Early stop check: Answer {key} found (AvgLogprob: {avg_logprob:.4f}). Counter updated: {key} -> {early_stop_answer_counter[key]}"
                                )
                                if early_stop_answer_counter[key] >= early_stop_threshold:
                                    early_stopped_answer = key
                                    early_stop_triggered = True
                                    print(
                                        f"\n!!! Early stopping triggered! Answer {key} reached count {early_stop_answer_counter[key]} (>= {early_stop_threshold}) !!!"
                                    )
                                    # Break inner loop (processing 'done' tasks)
                                    break
                            else:
                                print(
                                    f"  Early stop check: Non-whole integer answer '{boxed_text}' (AvgLogprob: {avg_logprob:.4f}), skipping."
                                )
                        except Exception as e:
                            print(
                                f"Error during early stopping check for index {index}, answer '{boxed_text}' (AvgLogprob: {avg_logprob:.4f}): {e}"
                            )

            except asyncio.CancelledError:
                done_tasks.add(task)
                newly_completed += 1
                print(f"Task {index} was cancelled.")
            except Exception as e:
                print(f"Error retrieving result for task {index}: {e}")
                done_tasks.add(task)  # Mark as processed even on error
                newly_completed += 1

        if newly_completed > 0:
            pbar.update(newly_completed)
            tasks_processed_count += newly_completed

        pending_tasks = pending_tasks - done_tasks  # Correctly update pending tasks

        # Break outer loop if early stopping was triggered in the inner loop
        if early_stop_triggered:
            break

        # Status and timeout logic
        current_elapsed = time.time() - start_time_batch
        num_actually_completed = len(task_results)
        completed_percentage = (
            num_actually_completed / total_tasks_count * 100 if total_tasks_count > 0 else 0
        )

        # Correct progress bar if needed
        if pbar.n != tasks_processed_count:
            pbar.n = tasks_processed_count
            pbar.refresh()

        # Provide regular status updates
        if num_actually_completed >= last_status_count + 4 or (
            done and last_status_count == 0 and num_actually_completed > 0
        ):
            print(
                f"\nStatus: {num_actually_completed}/{total_tasks_count} tasks completed successfully ({completed_percentage:.1f}%). Total processed (incl. cancelled/error): {tasks_processed_count}. Elapsed time: {current_elapsed:.1f}s"
            )
            last_status_count = num_actually_completed - (num_actually_completed % 4)

        # --- Corrected Timeout Extension Logic ---
        remaining_time = current_max_time - current_elapsed
        if (
            remaining_time <= 5  # Check if remaining time is 5s or less
            and not extended_once  # Haven't extended yet
            and num_actually_completed < PRE_TIMEOUT_COMPLETION_PERC * total_tasks_count
        ):  # Less than PRE_TIMEOUT_COMPLETION_PERC completed
            # Extend timeout
            current_max_time += TIMEOUT_EXTENSION
            extended_once = True
            timeout_type = "extended"
            print(
                f"\nRemaining time <= 5s and less than {PRE_TIMEOUT_COMPLETION_PERC * 100:.1f}% tasks completed ({num_actually_completed}/{total_tasks_count}). Extending timeout by {TIMEOUT_EXTENSION}s. New total time limit: {current_max_time:.1f} seconds."
            )

        # Check if time limit reached
        if current_elapsed > current_max_time:
            print(
                f"\nTime limit reached ({current_max_time:.1f}s). Successfully completed: {num_actually_completed}/{total_tasks_count}. Cancelling remaining tasks."
            )
            break

    # Determine the reason for stopping the loop
    final_stop_reason = (
        "early_stopped"
        if early_stop_triggered
        else "timeout"
        if current_elapsed > current_max_time
        else "completed"
    )

    # Cancel any remaining tasks explicitly if loop broke due to timeout or early stopping
    if pending_tasks:
        print(f"Cancelling {len(pending_tasks)} remaining tasks due to {final_stop_reason}...")
        for task in pending_tasks:
            if not task.done():
                task.cancel()
                # Ensure cancelled tasks are also marked as 'processed' for the progress bar
                if task not in done_tasks:
                    done_tasks.add(task)
                    tasks_processed_count += 1
        # Wait briefly for cancellations to register (optional, but can help)
        await asyncio.sleep(0.1)

    # Ensure progress bar reflects final count (including cancelled/error)
    pbar.n = tasks_processed_count
    pbar.refresh()
    pbar.close()

    total_time_batch = time.time() - start_time_batch
    print(
        f"Batch finished ({final_stop_reason}). Completed successfully: {len(task_results)}/{total_tasks_count} tasks in {total_time_batch:.2f} seconds"
    )
    if early_stop_triggered:
        print(f"Early stopping answer: {early_stopped_answer}")

    # --- Process Completed Tasks ---
    completed_outputs_final = []
    completed_indices = []
    completed_params_used = []
    sort_keys_and_list_of_messages = []  # For completed tasks only

    sorted_completed_indices = sorted(task_results.keys())
    for index in sorted_completed_indices:
        outputs_list, task_params_info, time_taken, req_id, avg_logprob = task_results[index]
        if outputs_list:
            final_output = outputs_list[-1]
            completed_outputs_final.append(final_output)
            completed_indices.append(index)
            completed_params_used.append(task_params_info)

            # Process the output for logging
            generated_text = final_output.outputs[0].text
            num_tokens = len(final_output.outputs[0].token_ids)
            stop_reason = final_output.outputs[0].finish_reason  # Original finish reason
            extracted_answer = extract_boxed_text(generated_text)

            # Log the completed generation
            logger.add_generation_result(
                problem_id=problem_id,
                question=question,
                request_id=req_id,
                temperature=task_params_info["temp"],
                top_p=task_params_info["top_p"],
                num_tokens=num_tokens,
                time_taken=int(time_taken),
                generated_text=generated_text,
                stop_reason=stop_reason,  # Log the model's stop reason
                timeout_type=timeout_type,
                avg_logprob=avg_logprob,
                extracted_answer=extracted_answer,
                is_complete=True,  # These tasks completed successfully
            )

            # Create message history with assistant's response
            messages = original_list_of_messages[index].copy()
            messages.append({"role": "assistant", "content": generated_text})
            sort_keys_and_list_of_messages.append(
                (num_tokens, messages, task_params_info, avg_logprob)
            )

    # Print summary statistics for *completed* tasks
    num_completed = len(completed_outputs_final)
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN") and num_completed > 0:
        try:
            # Basic completion metrics
            print("\n" + "=" * 80)
            print(f"{'GENERATION SUMMARY (COMPLETED TASKS)':^80}")
            print("=" * 80)
            print(
                f"Tasks completed successfully: {num_completed}/{total_tasks_count} ({num_completed/total_tasks_count*100:.1f}%)"
            )
            print(f"Total time for batch: {total_time_batch:.2f} seconds")

            # Token statistics for completed tasks
            if completed_outputs_final:
                req_num_tokens = [
                    len(output.outputs[0].token_ids) for output in completed_outputs_final
                ]
                token_stats = pd.Series(req_num_tokens).describe()
                print(
                    f"\nToken statistics (completed): Min: {token_stats['min']:.0f} | Mean: {token_stats['mean']:.0f} | Max: {token_stats['max']:.0f}"
                )

                # Finish reasons for completed tasks
                req_finish_reason = [
                    output.outputs[0].finish_reason for output in completed_outputs_final
                ]
                print("\nFinish Reasons (completed):")
                for reason, count in pd.Series(req_finish_reason).value_counts().items():
                    print(f"  '{reason}': {count} ({count/len(req_finish_reason)*100:.1f}%)")

                avg_logprobs = [
                    lp
                    for _, _, _, lp in sort_keys_and_list_of_messages
                    if lp is not None and not math.isnan(lp)
                ]
                if avg_logprobs:
                    avg_logprob_stats = pd.Series(avg_logprobs).describe()
                    print(
                        f"\nAvg Logprob statistics (completed, valid): Min: {avg_logprob_stats['min']:.4f} | Mean: {avg_logprob_stats['mean']:.4f} | Max: {avg_logprob_stats['max']:.4f}"
                    )
                else:
                    print("\nNo valid avg_logprob values calculated for completed tasks.")
        except Exception as e:
            print(f"Error while calculating stats: {e}")

    # --- Process Incomplete/Cancelled Tasks ---
    incomplete_indices = set(range(total_tasks_count)) - set(completed_indices)
    # The reason for cancellation applies to all these tasks
    cancelled_reason = (
        "early_stopped"
        if early_stop_triggered
        else "timeout"
        if final_stop_reason == "timeout"
        else "unknown_cancel"
    )

    for index in incomplete_indices:
        temp, top_p = assigned_params[index]
        incomplete_text = ""
        incomplete_tokens = 0
        # Get partial text if available from the dictionary
        if index in partial_outputs_dict:
            last_output_before_stop = partial_outputs_dict[index]
            incomplete_text = last_output_before_stop.outputs[0].text
            incomplete_tokens = len(last_output_before_stop.outputs[0].token_ids)

        # Log the incomplete/cancelled generation
        logger.add_generation_result(
            problem_id=problem_id,
            question=question,
            request_id=f"cancelled_{index}",  # Indicate cancellation
            temperature=temp,
            top_p=top_p,
            num_tokens=incomplete_tokens,
            time_taken=int(total_time_batch),  # Approximation (full batch time)
            generated_text=incomplete_text,
            stop_reason=cancelled_reason,  # Log 'early_stopped' or 'timeout'
            timeout_type=timeout_type,  # Log whether timeout was extended
            avg_logprob=float("nan"),
            extracted_answer=extract_boxed_text(incomplete_text),
            is_complete=False,  # These tasks did not complete fully
        )

    # Prepare the final list of messages (only successfully completed ones, sorted by token length)
    if sort_keys_and_list_of_messages:
        sort_keys_and_list_of_messages.sort(key=lambda x: x[0])
        # Return (messages, avg_logprob) tuples
        final_list_of_messages = [
            (messages, avg_logprob)
            for _, messages, _, avg_logprob in sort_keys_and_list_of_messages
        ]
    else:
        final_list_of_messages = []

    # Save current batch of logs
    logger.save_to_csv()

    print(
        f"Returning {len(final_list_of_messages)} successfully completed message histories from batch_message_generate."
    )
    # Return completed messages AND the early stopped answer (or None)
    return final_list_of_messages, early_stopped_answer


def create_starter_messages(question, index):
    """Create a variety of system and user prompts for diverse generation attempts."""
    options = []
    for _ in range(13):
        options.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\\\boxed{}, after taking modulo 1000.",
                },
                {"role": "user", "content": question},
            ]
        )
    for _ in range(3):
        options.append(
            [
                {
                    "role": "system",
                    "content": 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. After you get your final answer, take modulo 1000, and return the final answer within \\\\boxed{}."',
                },
                {"role": "user", "content": question},
            ],
        )
    return options[index % len(options)]


async def predict_for_question(problem_id: str, question: str) -> int:
    """
    Main prediction function for a single problem.

    Args:
        problem_id: Unique identifier for the problem
        question: The problem text

    Returns:
        The final answer (integer)
    """

    selected_questions_only = False
    if selected_questions_only and not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        # return 210
        if "Fred and George" not in question:
            return 210

    if time.time() > cutoff_time:
        print(f"Cutoff time reached for problem {problem_id}: {question[:50]}...")
        return 210

    print(f"\n--- Processing Problem {problem_id}: {question[:100]}... ---")

    num_seqs = MAX_NUM_SEQS
    # Optional: Adjust num_seqs based on remaining time
    # if time.time() > cutoff_times[-1]:
    #     num_seqs = max(1, 2 * MAX_NUM_SEQS // 3) # Ensure at least 1 seq
    #     print(f"Reduced num_seqs to {num_seqs} due to time constraints.")

    initial_list_of_messages = [
        create_starter_messages(question, index) for index in range(num_seqs)
    ]

    # Generate responses and check for early stopping answer
    completed_messages, early_stopped_answer = await batch_message_generate(
        initial_list_of_messages, problem_id, question
    )

    # --- Determine Final Answer ---
    if early_stopped_answer is not None:
        print(
            f"--> Early stopping triggered. Using answer: {early_stopped_answer} for problem {problem_id}"
        )
        final_answer = early_stopped_answer
    else:
        print(f"--> No early stopping for problem {problem_id}. Proceeding with voting.")
        # Filter answers from the successfully completed messages
        _, extracted_answers = batch_message_filter(
            completed_messages
        )

        if not extracted_answers:
            print(
                f"  No answers extracted from {len(completed_messages)} completed sequences. Returning default 210."
            )
            final_answer = 210
        else:
            # Select the final answer using our voting mechanism on successfully completed answers
            final_answer = select_answer(extracted_answers)
            print(f"--> Final selected answer (voting) for problem {problem_id}: {final_answer}")

    print("------\n\n")

    if cutoff_times:  # Avoid error if list becomes empty
        cutoff_times.pop()

    return final_answer


# Use a single event loop for the entire application
MAIN_LOOP = None


def get_event_loop():
    global MAIN_LOOP
    if MAIN_LOOP is None:
        MAIN_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(MAIN_LOOP)
        print(f"Initialized asyncio event loop: {type(MAIN_LOOP)}")
    return MAIN_LOOP


def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Synchronous wrapper for predict function that interfaces with Kaggle's evaluation system"""
    problem_id = str(
        id_.item(0)
    )  # Convert to string explicitly to handle ids with scientific notation like 192e23
    print("------")
    print(f"Problem ID: {problem_id}")
    question_text = question.item(0)

    # Reuse the same event loop
    loop = get_event_loop()
    answer = loop.run_until_complete(predict_for_question(problem_id, question_text))

    print("------\n\n")
    return pl.DataFrame({"id": problem_id, "answer": answer})


# Setup for testing locally or in Kaggle environment
# pd.read_csv(eval_file).drop("answer", axis=1).head(6).to_csv("reference.csv", index=False)
pd.read_csv(eval_file).drop("answer", axis=1).to_csv("reference.csv", index=False) # all 30 aime 2025 problems


# Keep submit = True while experimenting on colab and kaggle interactive session
# and False while submitting (counter-intuitive but simple) so notebook will be saved quickly without model loading and inference
# but make sure everything runs fine during interactive session
# also pin correct torch_compile_cache dataset version
submit = True

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    # actual submission during rerun
    submit = True
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


# temp fix for colab 1xL4
# /usr/local/lib/python3.11/dist-packages/vllm/v1/engine/core_client.py L288
# self.ctx = zmq.asyncio.Context() if asyncio_mode else sync_ctx
#
# self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx
# Load model and start inference server if submitting
if submit:
    model_load_start_time = time.time()

    import vllm
    from vllm import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams

    print("vLLM:", vllm.__version__)

    # Common engine arguments for both v0 and v1
    common_engine_args = {
        "model": llm_model_pth,
        "disable_log_requests": True,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "trust_remote_code": True,
        "tensor_parallel_size": len(
            [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
        )
        or 4,
        "gpu_memory_utilization": 0.9,
        "seed": SEED,
        "load_format": "runai_streamer",
    }

    if os.environ["VLLM_USE_V1"] == "1":
        # Try to use compiled cache to speed up loading too risky!!
        # try:
        #     if torch_compile_cache_folder is not None and os.path.exists(torch_compile_cache_folder):
        #         cache_dir = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
        #         os.makedirs(cache_dir, exist_ok=True)
        #         print(f"Copying {torch_compile_cache_folder} to {cache_dir}...")
        #         shutil.copytree(torch_compile_cache_folder, cache_dir, dirs_exist_ok=True)
        #         print("Copy completed successfully.")
        # except Exception as e:
        #     print(f"Error: {e}")
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_args = AsyncEngineArgs(**common_engine_args)
        print(f"{engine_args=}")
        llm = AsyncLLM.from_engine_args(engine_args)
        tokenizer = llm.processor.tokenizer.tokenizer
    else:  # V0
        from vllm import AsyncLLMEngine

        # For V0, we include additional parameters specific to this version
        v0_engine_args = {
            **common_engine_args,
            # "speculative_model": "/kaggle/input/deepseek-r1-distill-qwen-1-5b-draft-awq",
            # "num_speculative_tokens": 5,  # num_scheduler_steps should be 1 when using spec decode
            # spec-tp=4: The input size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.
            # "speculative_draft_tensor_parallel_size": 1,  # default same as target model tp size and 1 not recommeded cuda OOM
            "num_scheduler_steps": 16,
            "enable_prefix_caching": True,
            "kv_cache_dtype": "fp8_e4m3", # accuracy decreases, but enables double num_seqs
            "calculate_kv_scales": True, # dynamic, important to alleviate accuracy drop from fp8 kv cache
            "max_seq_len_to_capture": MAX_MODEL_LEN,
        }
        llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**v0_engine_args))
        tokenizer = llm.engine.tokenizer.tokenizer
    model_load_end_time = time.time()
    print(f"Full model loading time: {int(model_load_end_time - model_load_start_time)} seconds")
    
    disable_metrics_logger = True
    if disable_metrics_logger or os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        # turn off logs from metrics.py
        try:
            metrics_logger = logging.getLogger("vllm.engine.metrics")
            metrics_logger.setLevel(logging.WARNING)
            print("Successfully set log level for 'vllm.engine.metrics' to WARNING. INFO logs from metrics.py will be suppressed.")
        except Exception as e:
            print(f"Warning: Could not set log level for 'vllm.engine.metrics': {e}")



if submit:
    # Start the inference server
    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(("reference.csv",))

# this cell 9861 secs, 165 mins
# overall notebook: 2h50mins


eval_df = pd.read_csv(eval_file)
sol_df = pd.read_csv(logger.csv_path)
# consider completed generations 
compl_df = sol_df[(sol_df["is_complete"] == True)]

accuracy = 0

for pid, group_df in compl_df.groupby("problem_id"):
    extracted_answers_post = [(answer, logprob) for answer, logprob in group_df[["extracted_answer", "avg_logprob"]].values]
    final_answer = select_answer(extracted_answers_post)
    correct_answer = eval_df.query("id == @pid")["answer"].item()
    print(f"{final_answer=}\t{correct_answer=}")
    accuracy += int(final_answer == correct_answer)

print(f"\n**********\nFinal accuracy: {accuracy} / {compl_df['problem_id'].nunique()}\n**********\n")


import os
import pandas as pd



def style_throughput(preds_path, gt_path):
    """
    Reads prediction and ground truth CSV files, computes throughput, accuracy metrics,
    and returns a styled dataframe with multiple visual indicators of performance.

    Parameters:
    preds_path (str): Path to the predictions CSV file.
    gt_path (str): Path to the ground truth CSV file.

    Returns:
    pd.io.formats.style.Styler: Styled dataframe with multiple visual indicators.
    """
    # Load data
    df_preds = pd.read_csv(preds_path)
    df_gt = pd.read_csv(gt_path)

    # Merge predictions with ground truth
    df_merged = pd.merge(df_preds, df_gt, left_on="problem_id", right_on="id")

    # Convert answer and extracted_answer to numeric
    df_merged["answer"] = pd.to_numeric(df_merged["answer"], errors="coerce").astype("Int64")
    df_merged["extracted_answer"] = pd.to_numeric(
        df_merged["extracted_answer"], errors="coerce"
    ).astype("Int64", errors="ignore")

    # Compute correctness and throughput
    df_merged["correct_answer"] = (
        df_merged["extracted_answer"].eq(df_merged["answer"]).fillna(False)
    )
    df_merged["throughput"] = df_merged["num_tokens"] / df_merged["time_taken"]

    # Aggregate statistics
    df_agg = df_merged.groupby(
        [
            "problem_id",
            "timeout_type",
            "temperature",
            "top_p",
            "stop_reason",
        ]
    ).agg(
        {
            "throughput": ["min", "mean", "max", "count"],
            "num_tokens": ["min", "mean", "max", "count"],
            "avg_logprob": ["min", "mean", "max"],
            "correct_answer": [
                "sum",
                "count",
            ],
        }
    )

    # Flatten multi-index column names
    df_agg.columns = ["_".join(col) for col in df_agg.columns]

    # Calculate accuracy
    df_agg["accuracy"] = df_agg["correct_answer_sum"] / df_agg["correct_answer_count"]

    # Function to apply multiple visual cues based on accuracy
    def style_by_accuracy(row):
        accuracy = row["accuracy"]
        if np.isnan(accuracy):
            accuracy = 0

        # 1. Font weight (bold)
        min_weight = 400  # Normal
        max_weight = 700  # Bold
        weight = min_weight + int((max_weight - min_weight) * accuracy)

        # 2. Row thickness
        base_thickness = 1  # For accuracy 0
        max_thickness = 5  # For accuracy 1
        thickness = base_thickness + (max_thickness - base_thickness) * accuracy

        # 3. Background opacity
        # Higher accuracy = more opaque background
        opacity = 0.3 + (0.7 * accuracy)

        # 4. Font size
        base_font = 100  # percentage
        max_font = 115  # percentage
        font_size = base_font + int((max_font - base_font) * accuracy)

        # 5. Text decoration - underscore all columns when accuracy > 0
        # 6. Color gradient from red (low accuracy) to green (high accuracy)
        styles = []
        for _ in row:
            # Create red to green color gradient based on accuracy
            r = int(255 * (1 - accuracy))
            g = int(255 * accuracy)
            b = 0

            style = (
                f'font-weight: {weight}; '
                f'border-bottom: {thickness}px solid #888; '
                f'background-color: rgba({r},{g},{b},{opacity}); '
                f'font-size: {font_size}%; '
                f'color: {"white" if accuracy < 0.5 else "black"}; '  # Adjust text color for readability
            )

            # Add underline to all columns when accuracy > 0
            if accuracy > 0:
                underline_thickness = 1 + int(3 * accuracy)  # 1px to 4px thickness
                style += f"text-decoration: underline {underline_thickness}px;"

            styles.append(style)

        return styles

    # Create a formatter for accuracy column to show percentage
    def accuracy_formatter(value):
        return f"{value:.1%}" if not pd.isna(value) else "N/A"

        # Create formatters for avg_logprob columns
        def logprob_formatter(value):
            return f"{value:.3f}" if pd.notna(value) else "N/A"

    # Apply all styles and formatting
    styled_df = (
        df_agg.style.format({"accuracy": accuracy_formatter})
        .apply(style_by_accuracy, axis=1)  # This will color all columns based on accuracy
        .background_gradient(
            cmap="RdYlGn", subset=["throughput_min", "throughput_mean", "throughput_max"], axis=0
        )
        .background_gradient(
            cmap="RdYlGn_r", subset=["num_tokens_min", "num_tokens_mean", "num_tokens_max"], axis=0
        )
        .background_gradient(
            cmap="RdYlGn_r",
            subset=["avg_logprob_min", "avg_logprob_mean", "avg_logprob_max"],
            axis=0,
        )
        .background_gradient(cmap="RdYlGn", subset=["correct_answer_sum", "accuracy"], axis=0)
        # Set text alignment to center
        .set_properties(**{"text-align": "center"})
        # Add bar charts in the accuracy column
        .bar(subset=["accuracy"], color="#5fba7d", vmin=0, vmax=1)
    )

    return styled_df

if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    try:
        sdf = style_throughput(logger.csv_path, eval_file)
        display(sdf)
    except Exception as e:
        print(f"Error: {e}")































