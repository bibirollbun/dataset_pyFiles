pip install vllm pandas polars numpy torch transformers kaggle_evaluation


import os
import gc
import time
import warnings
import re
from collections import Counter
import random

import pandas as pd
import polars as pl
import numpy as np

import torch
import kaggle_evaluation.aimo_2_inference_server
from vllm import LLM, SamplingParams

# --- Configuration ---

# Model Configuration
if os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # Path in the Kaggle environment - VERIFY THIS MATCHES YOUR INPUT DATASET
    LLM_MODEL_PATH = '/kaggle/input/m/huikang/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1'
else:
    # --- !! IMPORTANT !! ---
    # Set this to the path where your DeepSeek model is stored locally for testing
    LLM_MODEL_PATH = '/kaggle/input/m/huikang/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1' # CHANGE THIS FOR LOCAL RUNS
    if LLM_MODEL_PATH == '/kaggle/input/m/huikang/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1' or not os.path.exists(LLM_MODEL_PATH):
         print(f"Warning: Local model path not set or not found: {LLM_MODEL_PATH}. Local testing might fail.")
         # Point to default Kaggle path as fallback for structure checking
         LLM_MODEL_PATH = '/kaggle/input/m/huikang/deepseek-r1/transformers/deepseek-aideepseek-r1-distill-qwen-14b-awq-neody/1'


# vLLM Configuration
TENSOR_PARALLEL_SIZE = 4      # Number of GPUs (Set to 4 for TPU v3-8)
MAX_NUM_SEQS = 16             # Max sequences for vLLM to process in parallel
MAX_MODEL_LEN = 8192          # Model's context length
GPU_MEMORY_UTILIZATION = 0.95 # GPU memory fraction for vLLM

# Time Management Configuration
TOTAL_RUNTIME_LIMIT_SECONDS = (4 * 60 + 45) * 60 # 4h 45m (adjust if needed)
STARTUP_TIME_BUFFER_SECONDS = 15 * 60
PER_QUESTION_TIME_LIMIT_SECONDS = 30 * 60 # API limit (enforced by gateway)

# Ensemble Configuration
# Use MAX_NUM_SEQS for efficiency, ensuring vLLM batches are full
NUM_ENSEMBLE_SEQUENCES = MAX_NUM_SEQS
NUM_REGENERATION_ATTEMPTS = 1 # How many times to try generating if first fails

# --- Environment Setup ---
pd.set_option('display.max_colwidth', None)
warnings.simplefilter('ignore')

# Set environment variables for CUDA (vLLM handles this via tensor_parallel_size)
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(TENSOR_PARALLEL_SIZE))) # Not strictly needed for vLLM TP
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Timing Setup ---
start_time = time.time()
absolute_cutoff_time = start_time + TOTAL_RUNTIME_LIMIT_SECONDS
# Generate time cutoffs for potential strategy adjustments (e.g., reducing max_tokens)
time_intervals = np.linspace(absolute_cutoff_time - (15 * 60), start_time + (180 * 60), 50 + 1) # 50 steps
cutoff_times_per_question = sorted([int(x) for x in time_intervals], reverse=True) # Pop from end for earlier times

print(f"Script start time: {start_time}")
print(f"Absolute cutoff time: {absolute_cutoff_time}")

# --- Globals ---
processed_ids = set() # Keep track of processed IDs
first_prediction_done = False # Track if the first (untimed) prediction is complete


# %% [code]
# --- Model Loading ---
print("Loading LLM with vLLM...")
try:
    llm = LLM(
        model=LLM_MODEL_PATH,
        tokenizer=LLM_MODEL_PATH,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=2024,
        quantization='awq',
        # --- Explicitly set dtype to float16 ---
        dtype='float16' # or dtype=torch.float16
        # -----------------------------------------
    )
    tokenizer = llm.get_tokenizer()
    print("LLM and Tokenizer loaded successfully via vLLM.")
except Exception as e:
    print(f"FATAL: Error loading LLM with vLLM from {LLM_MODEL_PATH}: {e}")
    import traceback
    traceback.print_exc()
    raise e


# --- Helper Functions ---

def extract_boxed_answer(text: str) -> str | None:
    """
    Extracts the last valid \\boxed{...} answer from the text.
    Includes fallbacks for \[...] and $$...$$. Returns string content or None.
    """
    if not isinstance(text, str): return None
    # Primary: \\boxed{...}
    matches = re.findall(r'\\boxed{(.*?)}', text, re.DOTALL)
    if not matches: # Fallback 1: \[ ... \]
        matches = re.findall(r'\\\[(.*?)\\\]', text, re.DOTALL)
        if not matches: # Fallback 2: $$ ... $$
             matches = re.findall(r'\$\$(.*?)\$\$', text, re.DOTALL)
             if not matches: # Fallback 3: Original notebook pattern `oxed{...}`
                 matches = re.findall(r'oxed{(.*?)}', text, re.DOTALL)
                 if not matches: return None # No patterns found

    # Find the last non-empty match
    for match in matches[::-1]:
        cleaned_match = match.strip()
        if cleaned_match:
            # Remove commas for potential int conversion later
            if re.match(r'^-?\d{1,3}(?:,\d{3})*$', cleaned_match):
                 cleaned_match = cleaned_match.replace(',', '')
            return cleaned_match # Return the raw string content
    return None # All matches were empty

def select_final_answer(answers: list[str | None]) -> int:
    """
    Selects the final answer from a list of extracted strings using majority vote.
    Counts valid integers, performs majority voting with jitter, takes modulo 1000.
    """
    counter = Counter()
    valid_answers_found = False
    for answer_str in answers:
        if answer_str is None: continue
        try: # Try converting to float first (handles "10.0") then check if integer
            num_float = float(answer_str)
            if num_float == int(num_float):
                num_int = int(num_float)
                counter[num_int] += 1 + random.random() / 1000.0 # Add jitter for tie-breaking
                valid_answers_found = True
        except (ValueError, TypeError): pass # Ignore non-numeric strings

    if not valid_answers_found:
        print("Warning: No valid numerical answers found in ensemble. Returning 0.")
        return 0

    # Sort by frequency (descending due to jitter)
    sorted_answers = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    best_answer_int = sorted_answers[0][0] # Most frequent integer

    # Return the answer modulo 1000
    final_answer = best_answer_int % 1000
    return final_answer

def create_prompt_options(question: str) -> list[list[dict]]:
    """ Creates a list of different prompt structures for ensemble generation. """
    prompts = []
    # Define system prompts based on successful strategies
    sys_prompt1 = "You are a the most powerful math expert. Please solve the problems with deep reasoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."
    sys_prompt2 = "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."
    sys_prompt3 = "Please carefully read the problem statement first to ensure you fully understand its meaning and key points. Then, solve the problem correctly and completely through deep reasoning. Finally, return the result modulo 1000 and enclose it in \\boxed{} like \"After take the result modulo 1000, final answer is \\boxed{180}\"."

    # Allocate prompts (adjust ratios as needed)
    num_p1 = NUM_ENSEMBLE_SEQUENCES * 10 // 16 # ~62.5%
    num_p2 = NUM_ENSEMBLE_SEQUENCES * 4 // 16  # 25%
    num_p3 = NUM_ENSEMBLE_SEQUENCES - num_p1 - num_p2 # Remaining

    for _ in range(num_p1): prompts.append([{"role": "system", "content": sys_prompt1}, {"role": "user", "content": question}])
    for _ in range(num_p2): prompts.append([{"role": "system", "content": sys_prompt2}, {"role": "user", "content": question}])
    for _ in range(num_p3): prompts.append([{"role": "system", "content": sys_prompt3}, {"role": "user", "content": question}])

    # Ensure exact count if rounding caused issues
    while len(prompts) < NUM_ENSEMBLE_SEQUENCES:
        prompts.append([{"role": "system", "content": sys_prompt1}, {"role": "user", "content": question}])

    return prompts[:NUM_ENSEMBLE_SEQUENCES]


# --- Main Prediction Logic ---

def predict_for_question(question: str, id_str: str) -> int:
    """
    Generates predictions for a single question using vLLM ensemble,
    extracts answers, selects the best one, and manages time.
    """
    global first_prediction_done, cutoff_times_per_question

    current_time = time.time()
    if current_time > absolute_cutoff_time - (1 * 60): # 1 min safety buffer
        print(f"Warning: Approaching absolute cutoff time for ID {id_str}. Returning 0.")
        return 0

    # Adjust max_tokens based on remaining time if needed
    max_tokens = MAX_MODEL_LEN * 3 // 4 # Default: generous token limit
    if cutoff_times_per_question and current_time > cutoff_times_per_question[-1]:
        print(f"Warning: Passed time interval cutoff for ID {id_str}. Reducing max_tokens.")
        max_tokens = max(1024, MAX_MODEL_LEN // 2) # Reduce significantly
        cutoff_times_per_question.pop()

    print(f"\n--- Predicting for ID: {id_str} ---")
    print(f"Current Time: {current_time:.2f}, Max Tokens: {max_tokens}")

    list_of_messages = create_prompt_options(question)
    all_extracted_answers = []
    successful_generations = 0

    # --- Generation Loop (currently 1 attempt) ---
    for attempt in range(NUM_REGENERATION_ATTEMPTS):
        if not list_of_messages: break # Stop if no prompts left
        print(f"Generation attempt {attempt + 1}/{NUM_REGENERATION_ATTEMPTS} with {len(list_of_messages)} prompts for ID {id_str}.")

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            # min_p=0.01, # From original notebook
            skip_special_tokens=True,
            max_tokens=max_tokens,
        )

        # Prepare prompts in the format vLLM expects
        prompts_for_vllm = [
            tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            ) for messages in list_of_messages
        ]

        request_outputs = []
        try:
            # Generate responses in batch using vLLM
            request_outputs = llm.generate(
                prompts=prompts_for_vllm,
                sampling_params=sampling_params,
                use_tqdm=False # Cleaner logs
            )
            successful_generations += len(request_outputs)
        except Exception as e:
            print(f"Error during vLLM generation for ID {id_str}, attempt {attempt + 1}: {e}")
            import traceback
            traceback.print_exc() # Show full traceback for debugging

        generated_texts = [ro.outputs[0].text for ro in request_outputs]
        num_outputs_received = len(generated_texts)
        print(f"Received {num_outputs_received} outputs for ID {id_str}.")

        # --- Answer Extraction and Filtering ---
        current_batch_answers = []
        remaining_prompts_messages = [] # Store full message lists for potential retry

        original_indices = list(range(len(list_of_messages))) # Indices of prompts sent

        output_idx = 0
        for prompt_idx in original_indices:
            if output_idx < num_outputs_received:
                generated_text = generated_texts[output_idx]
                extracted = extract_boxed_answer(generated_text)
                current_batch_answers.append(extracted)

                # --- Local Debug Saving ---
                if not (os.getenv('KAGGLE_KERNEL_RUN_TYPE') or os.getenv('KAGGLE_IS_COMPETITION_RERUN')):
                    try:
                        debug_data = {"id": id_str, "prompt_style_approx": prompt_idx % 3, "attempt": attempt + 1,
                                      "generated_text": generated_text, "extracted_answer": extracted}
                        debug_filename = f"local_debug_outputs_deepseek.csv"
                        mode = 'a' if os.path.exists(debug_filename) else 'w'; header = mode == 'w'
                        pd.DataFrame([debug_data]).to_csv(debug_filename, mode=mode, header=header, index=False)
                    except Exception as debug_e: print(f"Local Debug Save Error: {debug_e}")
                # --- End Local Debug Saving ---

                # Keep prompt only if no answer extracted AND if retries are enabled (>1 attempt)
                if extracted is None and NUM_REGENERATION_ATTEMPTS > 1:
                    remaining_prompts_messages.append(list_of_messages[prompt_idx])
            else: # Handle missing outputs from vLLM
                print(f"Warning: No output received for prompt index {prompt_idx} (ID: {id_str}). Keeping prompt if retries enabled.")
                if NUM_REGENERATION_ATTEMPTS > 1:
                     remaining_prompts_messages.append(list_of_messages[prompt_idx])
            output_idx += 1

        all_extracted_answers.extend(current_batch_answers)
        list_of_messages = remaining_prompts_messages # Update list for potential next attempt

        # Optional memory cleanup
        del request_outputs, prompts_for_vllm, generated_texts
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Final Answer Selection ---
    print(f"Finished generation attempts for ID {id_str}.")
    print(f"Total successful generations: {successful_generations}")
    print(f"Extracted answers for {id_str}: {all_extracted_answers}")

    final_answer = select_final_answer(all_extracted_answers)
    print(f"Selected final answer for {id_str}: {final_answer}")
    print("-" * 30)

    if not first_prediction_done: first_prediction_done = True
    return final_answer


# --- Kaggle API predict function ---

# IMPORTANT: The gateway might send Series or DataFrames. Handle both.
def predict(id_: pl.Series | pl.DataFrame, problem: pl.Series | pl.DataFrame) -> pl.DataFrame:
    """ API function called by the gateway. Handles Series or DataFrame input. """
    global processed_ids

    try: # Use .item() for Series, standard indexing for DataFrame
        if isinstance(id_, pl.Series):
            id_str = id_.item()
        else: # Assume DataFrame
            id_str = id_['id'][0]

        if isinstance(problem, pl.Series):
            question_str = problem.item()
        else: # Assume DataFrame
            question_str = problem['problem'][0]

    except Exception as e:
        print(f"CRITICAL Error extracting id/problem string: {e}")
        # Log inputs for debugging if possible, structure might be unexpected
        print(f"Received id_ object type: {type(id_)}, content: {id_}")
        print(f"Received problem object type: {type(problem)}, content: {problem}")
        error_id_str = "extraction_error_id" # Fallback ID
        # Attempt to get ID even if problem fails, as last resort
        try:
           if isinstance(id_, pl.Series): error_id_str = id_.item()
           elif id_ is not None and 'id' in id_.columns and len(id_) > 0: error_id_str = id_['id'][0]
        except: pass
        return pl.DataFrame({'id': [error_id_str], 'answer': [0]}) # Return 0 on error

    print(f"Received request for ID: {id_str}")

    if id_str in processed_ids:
        print(f"Warning: Received duplicate ID {id_str}. Returning default answer 0.")
        return pl.DataFrame({'id': [id_str], 'answer': [0]})

    answer = predict_for_question(question_str, id_str)
    processed_ids.add(id_str)
    return pl.DataFrame({'id': [id_str], 'answer': [answer]})


# --- Prepare reference file for local testing ---
REFERENCE_CSV_PATH_INPUT = '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv'
TEST_CSV_PATH_INPUT = '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv' # Fallback
LOCAL_TEST_FILE = 'local_test_data_deepseek.csv' # Use a different name to avoid conflicts
input_csv_to_use = None

# This block only runs when not in the official scoring environment
if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None:
    print("Running PREPARATION block because KAGGLE_IS_COMPETITION_RERUN is not set.")
    print("Preparing data file for local testing...")

    # Check reference file existence
    print(f"Checking for reference file at: {REFERENCE_CSV_PATH_INPUT}")
    if os.path.exists(REFERENCE_CSV_PATH_INPUT):
        print(f"Found reference file: {REFERENCE_CSV_PATH_INPUT}")
        input_csv_to_use = REFERENCE_CSV_PATH_INPUT
    else:
        print(f"Reference file NOT found. Checking for test file at: {TEST_CSV_PATH_INPUT}")
        if os.path.exists(TEST_CSV_PATH_INPUT):
             print(f"Found test file: {TEST_CSV_PATH_INPUT}")
             input_csv_to_use = TEST_CSV_PATH_INPUT
        else: print(f"Warning: Neither reference CSV nor test CSV found at specified paths."); LOCAL_TEST_FILE = None

    if input_csv_to_use:
        try:
            print(f"Attempting to read: {input_csv_to_use}")
            df_local = pd.read_csv(input_csv_to_use)
            columns_to_keep = ['id', 'problem']
            if all(col in df_local.columns for col in columns_to_keep):
                print(f"Found required columns. Preparing {LOCAL_TEST_FILE}...")
                df_local[columns_to_keep].to_csv(LOCAL_TEST_FILE, index=False)
                if os.path.exists(LOCAL_TEST_FILE): print(f"Successfully created {LOCAL_TEST_FILE}.")
                else: print(f"ERROR: Failed to create {LOCAL_TEST_FILE}!"); LOCAL_TEST_FILE = None
            else: print(f"Error: Input CSV {input_csv_to_use} missing required columns."); LOCAL_TEST_FILE = None
        except Exception as e: print(f"Error during file preparation: {e}"); LOCAL_TEST_FILE = None
    else: print("No input CSV file found to prepare."); LOCAL_TEST_FILE = None

    if LOCAL_TEST_FILE and os.path.exists(LOCAL_TEST_FILE): print(f"Preparation successful. Local test will use: {LOCAL_TEST_FILE}")
    else: print(f"Preparation failed or skipped. Local test file variable is: {LOCAL_TEST_FILE}")
else:
    print("Skipping local file preparation because KAGGLE_IS_COMPETITION_RERUN is set.")
    LOCAL_TEST_FILE = None


# --- Server Initialization and Run ---

print("Initializing Inference Server...")
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN') is None:
    # Run locally using the prepared reference file
    print("Running in Local mode. Starting local gateway test...")
    if LOCAL_TEST_FILE and os.path.exists(LOCAL_TEST_FILE):
        try:
            inference_server.run_local_gateway( (LOCAL_TEST_FILE,) ) # Pass path as tuple
            print("Local gateway test finished successfully.")
        except Exception as local_e:
            print(f"Error during local gateway run: {local_e}")
            import traceback
            traceback.print_exc()
        finally: # Cleanup temporary file
            if os.path.exists(LOCAL_TEST_FILE):
                try: os.remove(LOCAL_TEST_FILE); print(f"Cleaned up {LOCAL_TEST_FILE}")
                except OSError as e: print(f"Error removing {LOCAL_TEST_FILE}: {e}")
    else:
        print("Local test file not available/prepared. Skipping local gateway run.")
else:
    # Run in Kaggle scoring environment
    print("Running in Kaggle Rerun mode. Starting server to wait for requests...")
    inference_server.serve() # Blocks until completion

print("Script execution finished.")

