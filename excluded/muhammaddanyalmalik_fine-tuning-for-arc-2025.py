# # English: Use jq to open the JSON file and convert it into key-value entries
# # Ø§Ø±Ø¯Ùˆ: jq Ø§Ø³ØªØ¹Ù…Ø§Ù„ Ú©Ø± Ú©Û’ JSON Ù�Ø§Ø¦Ù„ Ú©Ùˆ key-value entries Ù…ÛŒÚº ØªØ¨Ø¯ÛŒÙ„ Ú©Ø±Ù†Ø§
# echo "Step 1: Converting JSON to entries..."
# !jq 'to_entries' /kaggle/input/arc24-source-code/new_partitions/val_rs7.json > step1.json

# # English: Select only the first 2 entries from the JSON
# # Ø§Ø±Ø¯Ùˆ: JSON Ø³Û’ ØµØ±Ù� Ù¾Û�Ù„Û’ Ø¯Ùˆ entries Ù…Ù†ØªØ®Ø¨ Ú©Ø±Ù†Ø§
# echo "Step 2: Selecting first 2 entries..."
# !jq '.[:2]' step1.json > step2.json

# # English: Convert the selected entries back into JSON object format
# # Ø§Ø±Ø¯Ùˆ: Ù…Ù†ØªØ®Ø¨ Ú©ÛŒ Ú¯Ø¦ÛŒ entries Ú©Ùˆ Ø¯ÙˆØ¨Ø§Ø±Û� JSON object Ù…ÛŒÚº ØªØ¨Ø¯ÛŒÙ„ Ú©Ø±Ù†Ø§
# echo "Step 3: Converting back to JSON object..."
# !jq 'from_entries' step2.json > smaller_val_challenges.json

# # English: Final file created successfully
# # Ø§Ø±Ø¯Ùˆ: Ø¢Ø®Ø±ÛŒ Ù�Ø§Ø¦Ù„ Ú©Ø§Ù…ÛŒØ§Ø¨ÛŒ Ø³Û’ Ø¨Ù† Ú¯Ø¦ÛŒ
# echo "âœ… smaller_val_challenges.json created successfully!"

!jq 'to_entries | .[:2] | from_entries' /kaggle/input/arc24-source-code/new_partitions/val_rs7.json > smaller_val_challenges.json


# ------------------------------
# Training & Splitting Settings
# ------------------------------

n_splits = 100  # English: Number of splits (options: 2, 4, 10, 20, 50, 100)
                # Ø§Ø±Ø¯Ùˆ: ÚˆÛŒÙ¹Ø§ Ú©Ùˆ Ú©ØªÙ†Û’ Ø­ØµÙˆÚº Ù…ÛŒÚº ØªÙ‚Ø³ÛŒÙ… Ú©Ø±Ù†Ø§ Û�Û’

total_train_steps = 32000  # English: Total training steps
                          # Ø§Ø±Ø¯Ùˆ: Ú©Ù„ Ù¹Ø±ÛŒÙ†Ù†Ú¯ Ú©Û’ steps

# ------------------------------
# Model Configuration
# ------------------------------
class cfg:
    # English: Path of the main model
    # Ø§Ø±Ø¯Ùˆ: Ù…ÛŒÙ† Ù…Ø§ÚˆÙ„ Ú©Ø§ Ø±Ø§Ø³ØªÛ�
    model_path = '/kaggle/input/qwen2.5/transformers/0.5b-instruct/1'
    
    # English: Path of LoRA (Low-Rank Adaptation)
    # Ø§Ø±Ø¯Ùˆ: LoRA Ù…Ø§ÚˆÙ„ Ú©Ø§ Ø±Ø§Ø³ØªÛ�
    input_lora_path = '/kaggle/input/loras/transformers/qwen2.5-0.5b-instruct/8'
    
    # English: Which prompt version to use
    # Ø§Ø±Ø¯Ùˆ: Ú©ÙˆÙ† Ø³Ø§ Ù¾Ø±Ø§Ù…Ù¾Ù¹ ÙˆØ±Ú˜Ù† Ø§Ø³ØªØ¹Ù…Ø§Ù„ Û�ÙˆÚ¯Ø§
    prompt_version = 'output-from-examples-v1'
    
    # English: Where to save the merged model
    # Ø§Ø±Ø¯Ùˆ: Ù…Ø±Ø¬ Û�ÙˆÙ†Û’ ÙˆØ§Ù„Ø§ Ù…Ø§ÚˆÙ„ Ú©Û�Ø§Úº Ù…Ø­Ù�ÙˆØ¸ Û�ÙˆÚ¯Ø§
    merged_model_path = '/kaggle/tmp/qwen_merged_model'
    
    # English: Encoder type for processing grid-based inputs
    # Ø§Ø±Ø¯Ùˆ: grid Ø§Ù† Ù¾Ù¹ Ú©Ùˆ Ù¾Ø±Ø§Ø³ÛŒØ³ Ú©Ø±Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ encoder Ú©ÛŒ Ù‚Ø³Ù…
    grid_encoder = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    
    max_model_len = 10240  # English: Max length of the model input
                           # Ø§Ø±Ø¯Ùˆ: Ù…Ø§ÚˆÙ„ Ø§Ù† Ù¾Ù¹ Ú©ÛŒ Ø²ÛŒØ§Ø¯Û� Ø³Û’ Ø²ÛŒØ§Ø¯Û� Ù„Ù…Ø¨Ø§Ø¦ÛŒ
    
    # ------------------------------
    # Dataset Configuration
    # ------------------------------
    dataset_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    # dataset_path = '/kaggle/input/arc25-source-code/new_partitions/val_rs7.json'
    # dataset_path = 'smaller_val_challenges.json'
    
    split_size = 100 // n_splits  # English: Tasks per split
                                  # Ø§Ø±Ø¯Ùˆ: Û�Ø± Ø­ØµÛ’ Ù…ÛŒÚº Ú©ØªÙ†Û’ Ù¹Ø§Ø³Ú© Û�ÙˆÚº Ú¯Û’
    
    # ------------------------------
    # Fine-tuning Parameters
    # ------------------------------
    max_steps = total_train_steps // n_splits  # English: Steps per split
                                               # Ø§Ø±Ø¯Ùˆ: Û�Ø± Ø­ØµÛ’ Ù…ÛŒÚº steps
    
    learning_rate = 8e-5  # English: Learning rate
                          # Ø§Ø±Ø¯Ùˆ: Ù„Ø±Ù†Ù†Ú¯ Ø±ÛŒÙ¹
    
    lr_scheduler_type: str = "linear"  # English: Learning rate scheduler
                                       # Ø§Ø±Ø¯Ùˆ: Ù„Ø±Ù†Ù†Ú¯ Ø±ÛŒÙ¹ Ú©Ùˆ Ú©Ù†Ù¹Ø±ÙˆÙ„ Ú©Ø±Ù†Û’ Ú©Ø§ Ø·Ø±ÛŒÙ‚Û�
    
    batch_size = 1  # English: Number of samples per batch
                    # Ø§Ø±Ø¯Ùˆ: Û�Ø± Ø¨ÛŒÚ† Ù…ÛŒÚº Ù†Ù…ÙˆÙ†ÙˆÚº Ú©ÛŒ ØªØ¹Ø¯Ø§Ø¯
    
    max_seq_len = 5120  # English: Maximum sequence length
                        # Ø§Ø±Ø¯Ùˆ: Ø²ÛŒØ§Ø¯Û� Ø³Û’ Ø²ÛŒØ§Ø¯Û� sequence Ù„Ù…Ø¨Ø§Ø¦ÛŒ
    
    # ------------------------------
    # Inference Parameters
    # ------------------------------
    predictions_per_task = 96  # English: Predictions per task (must be multiple of 8)
                               # Ø§Ø±Ø¯Ùˆ: Û�Ø± Ù¹Ø§Ø³Ú© Ú©Û’ Ù„ÛŒÛ’ Ú©ØªÙ†ÛŒ Ù¾Ø±ÛŒÚˆÚ©Ø´Ù†Ø²
    
    inference_timeout = "12m"  # English: Timeout for each split during inference
                               # Ø§Ø±Ø¯Ùˆ: Û�Ø± Ø­ØµÛ’ Ú©Ø§ Ø²ÛŒØ§Ø¯Û� Ø³Û’ Ø²ÛŒØ§Ø¯Û� ÙˆÙ‚Øª
    
    # ------------------------------
    # Ensemble Configuration
    # ------------------------------
    ensemble_with_2020: bool = True  # English: Combine with 2025 model outputs
                                     # Ø§Ø±Ø¯Ùˆ: 2020 Ù…Ø§ÚˆÙ„ Ú©Û’ Ù†ØªØ§Ø¦Ø¬ Ú©Ùˆ Ø¨Ú¾ÛŒ Ø´Ø§Ù…Ù„ Ú©Ø±ÛŒÚº


# ------------------------------
# Dry Run Check
# ------------------------------
import os

is_dry_run = (
    cfg.dataset_path == '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    and not os.getenv('KAGGLE_IS_COMPETITION_RERUN')
)

if is_dry_run:
    print("ğŸ’¡ Dry Run Mode: No inference or extra package installation will be done.")
else:
    print("ğŸš€ Running in Normal Mode: Inference and training will be executed.")


# ------------------------------
# Assertion Checks
# ------------------------------
if int(cfg.input_lora_path.split('/')[-1]) < 18 and cfg.input_lora_path.startswith('/kaggle/input/loras/transformers/qwen2-0.5b'):
    # English: If LoRA version is less than 18, use v0 prompts
    # Ø§Ø±Ø¯Ùˆ: Ø§Ú¯Ø± LoRA ÙˆØ±Ú˜Ù† 18 Ø³Û’ Ú©Ù… Û�Û’ ØªÙˆ v0 Ù¾Ø±Ø§Ù…Ù¾Ù¹ Ø§Ø³ØªØ¹Ù…Ø§Ù„ Û�ÙˆÚ¯Ø§
    assert cfg.prompt_version == 'output-from-examples-v0'
    print("âœ… Using prompt version: v0")
else:
    # English: Otherwise, use v1 prompts
    # Ø§Ø±Ø¯Ùˆ: ÙˆØ±Ù†Û� v1 Ù¾Ø±Ø§Ù…Ù¾Ù¹ Ø§Ø³ØªØ¹Ù…Ø§Ù„ Û�ÙˆÚ¯Ø§
    assert cfg.prompt_version == 'output-from-examples-v1'
    print("âœ… Using prompt version: v1")


# ------------------------------
# Import Required Libraries
# ------------------------------

import logging      # English: For logging events
                    # Ø§Ø±Ø¯Ùˆ: Ù„Ø§Ú¯Ù†Ú¯ Ú©Û’ Ù„ÛŒÛ’ (Ø§ÛŒÙˆÙ†Ù¹Ø³ Ø±ÛŒÚ©Ø§Ø±Úˆ Ú©Ø±Ù†Û’ Ú©Û’ Ù„ÛŒÛ’)

import subprocess   # English: To run system commands inside Python
                    # Ø§Ø±Ø¯Ùˆ: Ø³Ø³Ù¹Ù… Ú©Ù…Ø§Ù†ÚˆØ² Ú©Ùˆ Python Ø³Û’ Ú†Ù„Ø§Ù†Û’ Ú©Û’ Ù„ÛŒÛ’

import sys          # English: To modify system paths and settings
                    # Ø§Ø±Ø¯Ùˆ: Ø³Ø³Ù¹Ù… Ú©Û’ Ø±Ø§Ø³ØªÛ’ Ø§ÙˆØ± Ø³ÛŒÙ¹Ù†Ú¯Ø² Ø¨Ø¯Ù„Ù†Û’ Ú©Û’ Ù„ÛŒÛ’

import json         # English: For reading/writing JSON files
                    # Ø§Ø±Ø¯Ùˆ: JSON Ù�Ø§Ø¦Ù„Ø² Ù¾Ú‘Ú¾Ù†Û’ Ø§ÙˆØ± Ù„Ú©Ú¾Ù†Û’ Ú©Û’ Ù„ÛŒÛ’

import glob         # English: For finding files with patterns
                    # Ø§Ø±Ø¯Ùˆ: Ù�Ø§Ø¦Ù„Ø² ÚˆÚ¾ÙˆÙ†ÚˆÙ†Û’ Ú©Û’ Ù„ÛŒÛ’ (Ù¾Û�Ú†Ø§Ù† Ú©Û’ Ø³Ø§ØªÚ¾)

import os           # English: For interacting with the operating system
                    # Ø§Ø±Ø¯Ùˆ: Ø¢Ù¾Ø±ÛŒÙ¹Ù†Ú¯ Ø³Ø³Ù¹Ù… Ú©Û’ Ø³Ø§ØªÚ¾ Ú©Ø§Ù… Ú©Ø±Ù†Û’ Ú©Û’ Ù„ÛŒÛ’

import shutil       # English: For file/folder copy, move, delete operations
                    # Ø§Ø±Ø¯Ùˆ: Ù�Ø§Ø¦Ù„/Ù�ÙˆÙ„ÚˆØ± Ú©Ø§Ù¾ÛŒØŒ Ù…ÙˆÙˆ Ø§ÙˆØ± ÚˆÛŒÙ„ÛŒÙ¹ Ú©Û’ Ù„ÛŒÛ’

from tqdm.auto import tqdm  # English: For progress bars
                            # Ø§Ø±Ø¯Ùˆ: Ù¾Ø±ÙˆÚ¯Ø±ÛŒØ³ Ø¨Ø§Ø± Ø¯Ú©Ú¾Ø§Ù†Û’ Ú©Û’ Ù„ÛŒÛ’

print("âœ… Libraries imported successfully.")


# ------------------------------
# Add ARC24 source code to sys.path if not dry run
# ------------------------------

if not is_dry_run:
    sys.path.append('/kaggle/input/arc25-source-code')
    print("ğŸ“‚ Path added: /kaggle/input/arc25-source-code")
else:
    print("ğŸ’¡ Dry run mode: Skipping sys.path modification.")


# ------------------------------
# Configure Logging
# ------------------------------
# English: Logging helps in tracking events with timestamps
# Ø§Ø±Ø¯Ùˆ: Ù„Ø§Ú¯Ù†Ú¯ ÙˆÙ‚Øª Ú©Û’ Ø³Ø§ØªÚ¾ Ø§ÛŒÙˆÙ†Ù¹Ø³ Ú©Ùˆ Ù¹Ø±ÛŒÚ© Ú©Ø±Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ Ø§Ø³ØªØ¹Ù…Ø§Ù„ Û�ÙˆØªÛŒ Û�Û’

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

print("ğŸ“� Logging configured successfully.")


# # ------------------------------
# # Launch 2020 Solution in Background
# # ------------------------------

# if not is_dry_run and cfg.ensemble_with_2020:
#     # English: Notify that the 2020 solution is being launched
#     # Ø§Ø±Ø¯Ùˆ: Ø§Ø·Ù„Ø§Ø¹ Ø¯ÛŒÙ†Ø§ Ú©Û� 2020 Ú©Ø§ Ø­Ù„ Ø¨ÛŒÚ© Ú¯Ø±Ø§Ø¤Ù†Úˆ Ù…ÛŒÚº Ú†Ù„Ø§ÛŒØ§ Ø¬Ø§ Ø±Û�Ø§ Û�Û’
#     print("ğŸš€ Launching 2020 solution in the background...")

#     # English: Define the command-line arguments for running the 2020 solution
#     # Ø§Ø±Ø¯Ùˆ: 2020 Ø­Ù„ Ú©Ùˆ Ú†Ù„Ø§Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ Ú©Ù…Ø§Ù†Úˆ Ù„Ø§Ø¦Ù† arguments
#     args = [
#         # 'taskset', '-c', '0',  # English: Restrict job to a single CPU (commented out)
#         # Ø§Ø±Ø¯Ùˆ: ØµØ±Ù� Ø§ÛŒÚ© CPU Ù¾Ø± Ø¬Ø§Ø¨ Ú©Ùˆ Ù…Ø­Ø¯ÙˆØ¯ Ú©Ø±Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ (Ù�ÛŒ Ø§Ù„Ø­Ø§Ù„ disable Û�Û’)

#         'python',
#         '/kaggle/input/arc24-source-code/full_2020_solution.py',
#         f'--dataset_filepath={cfg.dataset_path}',
#         '--icecuber_output_filepath=icecuber_submission.json',
#         '--dsl_output_filepath=submission_program_search.json'
#     ]

#     # English: Start the process in the background, capturing output and errors
#     # Ø§Ø±Ø¯Ùˆ: Ø¨ÛŒÚ© Ú¯Ø±Ø§Ø¤Ù†Úˆ Ù…ÛŒÚº Ù¾Ø±Ø§Ø³ÛŒØ³ Ø´Ø±ÙˆØ¹ Ú©Ø±ÛŒÚº Ø§ÙˆØ± Ø¢Ø¤Ù¹ Ù¾Ù¹/Ø§ÛŒØ±Ø±Ø² Ú©Ùˆ Ø±ÛŒÚ©Ø§Ø±Úˆ Ú©Ø±ÛŒÚº
#     full_2020_solution_process = subprocess.Popen(
#         args,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )

#     print("âœ… 2020 solution process started successfully.")
# else:
#     print("ğŸ’¡ Skipping 2020 solution (either dry run or ensemble disabled).")

if not is_dry_run and cfg.ensemble_with_2020:
    print('Launching 2020 solution in the background')
    args = [
        #'taskset', '-c', '0', apparently this will restrict the job to a single cpu
        'python',
        '/kaggle/input/arc24-source-code/full_2020_solution.py',
        f'--dataset_filepath={cfg.dataset_path}',
        '--icecuber_output_filepath=icecuber_submission.json',
        '--dsl_output_filepath=submission_program_search.json']
    full_2020_solution_process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


%%time
if not is_dry_run:
    !bash /kaggle/input/arc24-source-code/install_libraries.sh


if not is_dry_run:
    from arc24.utils import ResourceMonitor
    monitor = ResourceMonitor(interval=1)
    monitor.start()


# ------------------------------
# Create Single Task Datasets
# ------------------------------
if not is_dry_run:
    single_task_datasets_path = 'single_task_datasets'

    # English: Create a folder for saving single task datasets
    # Ø§Ø±Ø¯Ùˆ: Ø³Ù†Ú¯Ù„ Ù¹Ø§Ø³Ú© ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹Ø³ Ù…Ø­Ù�ÙˆØ¸ Ú©Ø±Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ Ù�ÙˆÙ„ÚˆØ± Ø¨Ù†Ø§Ø¦ÛŒÚº
    os.makedirs(single_task_datasets_path, exist_ok=True)
    print(f"ğŸ“‚ Directory created/verified: {single_task_datasets_path}")

    # English: Load the dataset from the configured path
    # Ø§Ø±Ø¯Ùˆ: Ú©Ù†Ù�ÛŒÚ¯Ø±Úˆ Ø±Ø§Ø³ØªÛ’ Ø³Û’ ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹ Ù„ÙˆÚˆ Ú©Ø±ÛŒÚº
    with open(cfg.dataset_path, 'r') as f:
        items = list(json.load(f).items())
    print(f"ğŸ“¥ Loaded dataset with {len(items)} items.")

    # English: Ensure dataset can be divided evenly by split size
    # Ø§Ø±Ø¯Ùˆ: Ú†ÛŒÚ© Ú©Ø±ÛŒÚº Ú©Û� ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹ Ú©Ùˆ Ø¨Ø±Ø§Ø¨Ø± Ø­ØµÙˆÚº Ù…ÛŒÚº ØªÙ‚Ø³ÛŒÙ… Ú©ÛŒØ§ Ø¬Ø§ Ø³Ú©ØªØ§ Û�Û’
    assert len(items) % cfg.split_size == 0
    print(f"âœ… Dataset is divisible by split size ({cfg.split_size}).")

    # English: Create single-task datasets by splitting
    # Ø§Ø±Ø¯Ùˆ: ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹ Ú©Ùˆ Ø­ØµÙˆÚº Ù…ÛŒÚº ØªÙ‚Ø³ÛŒÙ… Ú©Ø± Ú©Û’ Ø³Ù†Ú¯Ù„ Ù¹Ø§Ø³Ú© ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹ Ø¨Ù†Ø§Ø¦ÛŒÚº
    for batch_idx in tqdm(range(len(items)//cfg.split_size), desc='Creating single task datasets'):
        data = dict(items[batch_idx*cfg.split_size: (batch_idx + 1)*cfg.split_size])
        assert len(data) == cfg.split_size

        task_id = list(data.keys())[0]  # English: Take the first task ID
                                        # Ø§Ø±Ø¯Ùˆ: Ù¾Û�Ù„Ø§ Ù¹Ø§Ø³Ú© Ø¢Ø¦ÛŒ ÚˆÛŒ Ù„ÛŒÚº
        filepath = os.path.join(single_task_datasets_path, f'{task_id}.json')

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"ğŸ“� Created single-task dataset: {filepath}")

    # English: List generated files for confirmation
    # Ø§Ø±Ø¯Ùˆ: Ø¨Ù†Ø§Ø¦ÛŒ Ú¯Ø¦ÛŒ Ù�Ø§Ø¦Ù„Ø² Ú©Ùˆ Ø¯Ú©Ú¾Ø§Ø¦ÛŒÚº
    ! ls -lh {single_task_datasets_path}


# ------------------------------
# Create Training Datasets for TTFT
# ------------------------------
if not is_dry_run:
    training_datasets_path = 'single_task_training_datasets'

    # English: Create folder for training datasets
    # Ø§Ø±Ø¯Ùˆ: Ù¹Ø±ÛŒÙ†Ù†Ú¯ ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹Ø³ Ú©Û’ Ù„ÛŒÛ’ Ù�ÙˆÙ„ÚˆØ± Ø¨Ù†Ø§Ø¦ÛŒÚº
    os.makedirs(training_datasets_path, exist_ok=True)
    print(f"ğŸ“‚ Directory created/verified: {training_datasets_path}")

    # English: Get all single-task dataset files
    # Ø§Ø±Ø¯Ùˆ: ØªÙ…Ø§Ù… Ø³Ù†Ú¯Ù„ Ù¹Ø§Ø³Ú© ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹ Ù�Ø§Ø¦Ù„Ø² Ø­Ø§ØµÙ„ Ú©Ø±ÛŒÚº
    dataset_filepaths = glob.glob(os.path.join(single_task_datasets_path, '*.json'))
    print(f"ğŸ“¥ Found {len(dataset_filepaths)} single-task dataset files.")

    # English: Generate N-1 training datasets using script
    # Ø§Ø±Ø¯Ùˆ: Ø§Ø³Ú©Ø±Ù¾Ù¹ Ú©Û’ Ø°Ø±ÛŒØ¹Û’ N-1 Ù¹Ø±ÛŒÙ†Ù†Ú¯ ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹ Ø¨Ù†Ø§Ø¦ÛŒÚº
    for dataset_filepath in tqdm(dataset_filepaths, desc='Creating ttft training datasets'):
        print(f"âš™ï¸� Processing: {dataset_filepath}")
        !python /kaggle/input/arc24-source-code/create_n-1_dataset.py \
        {dataset_filepath} \
        {os.path.join(training_datasets_path, os.path.basename(dataset_filepath))}
        print(f"âœ… Training dataset created for: {dataset_filepath}")


%%time
def clean_train_output_except_adapter(output_dir):
    """
    English: Clean unnecessary files from training outputs to save disk space. 
    Urdu: ÚˆØ³Ú© Ø§Ø³Ù¾ÛŒØ³ Ø¨Ú†Ø§Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ Ù¹Ø±ÛŒÙ†Ù†Ú¯ Ø¢Ø¤Ù¹ Ù¾Ù¹ Ø³Û’ ØºÛŒØ± Ø¶Ø±ÙˆØ±ÛŒ Ù�Ø§Ø¦Ù„Ø² ØµØ§Ù� Ú©Ø±ÛŒÚºÛ”
    
    Note:
    - Max Disk is 57.6 GiB, but ~8 GiB are already used.
    - Each checkpoint is ~265 MB â†’ 26 GB for 100 savings.
    - '/kaggle/working/' allows only 20 GB.
    - A previous run failed due to disk error with 50 splits (~400 MB each).
    """
    print(f"ğŸ§¹ Cleaning unnecessary files in: {output_dir}")

    # English: Remove large and redundant files from checkpoints
    # Ø§Ø±Ø¯Ùˆ: Ø¨Ú‘ÛŒ Ø§ÙˆØ± ØºÛŒØ± Ø¶Ø±ÙˆØ±ÛŒ Ù�Ø§Ø¦Ù„Ø² Ú©Ùˆ Ú†ÛŒÚ© Ù¾ÙˆØ§Ø¦Ù†Ù¹Ø³ Ø³Û’ Û�Ù¹Ø§ Ø¯ÛŒÚº
    !rm {output_dir}/*/*.pth {output_dir}/*/*.pt {output_dir}/*/*.md {output_dir}/*/*.txt {output_dir}/*/*.bin {output_dir}/*/token*
    !rm {output_dir}/*/added_tokens.json {output_dir}/*/special_tokens_map.json {output_dir}/*/vocab.json {output_dir}/*/trainer_state.json

    print(f"âœ… Cleanup completed for: {output_dir}")


# ------------------------------
# Fine-Tuning Loop
# ------------------------------
if not is_dry_run:
    dataset_filepaths = sorted(glob.glob(os.path.join(training_datasets_path, '*.json')))
    checkpoints_folder = '/kaggle/tmp/checkpoints'  # English: Temp folder for checkpoints
                                                    # Ø§Ø±Ø¯Ùˆ: Ú†ÛŒÚ© Ù¾ÙˆØ§Ø¦Ù†Ù¹Ø³ Ú©Û’ Ù„ÛŒÛ’ Ø¹Ø§Ø±Ø¶ÛŒ Ù�ÙˆÙ„ÚˆØ±
    
    os.makedirs(checkpoints_folder, exist_ok=True)
    print(f"ğŸ“‚ Checkpoints folder created/verified: {checkpoints_folder}")
    print(f"ğŸ“¥ Found {len(dataset_filepaths)} training datasets to fine-tune.")

    # English: Loop through all datasets and fine-tune separately
    # Ø§Ø±Ø¯Ùˆ: ØªÙ…Ø§Ù… ÚˆÛŒÙ¹Ø§Ø³ÛŒÙ¹Ø³ Ù¾Ø± Ø§Ù„Ú¯ Ø§Ù„Ú¯ Ù�Ø§Ø¦Ù† Ù¹ÛŒÙˆÙ†Ù†Ú¯ Ú©Ø±ÛŒÚº
    for idx, dataset_filepath in enumerate(tqdm(dataset_filepaths, desc='Finetuning models'), start=1):
        task_name = os.path.splitext(os.path.basename(dataset_filepath))[0]
        output_dir = os.path.join(checkpoints_folder, task_name)

        print(f"ğŸš€ Starting fine-tuning {idx}/{len(dataset_filepaths)} â†’ {task_name}")

        # Run fine-tuning script
        !python /kaggle/input/arc24-source-code/fine-tuning.py \
        --model_path={cfg.model_path} \
        --adapter_path={cfg.input_lora_path} \
        --output_dir={output_dir} \
        --train_datasets {dataset_filepath} {cfg.prompt_version} \
        --val_dataset {dataset_filepath} {cfg.prompt_version} \
        --max_steps={cfg.max_steps} \
        --eval_steps={cfg.max_steps*2} \
        --max_seq_len={cfg.max_seq_len} \
        --learning_rate={cfg.learning_rate} \
        --lr_scheduler_type={cfg.lr_scheduler_type} \
        --batch_size={cfg.batch_size} \
        --report_to=tensorboard \
        --grid_encoder="{cfg.grid_encoder}" \
        --remove_train_samples_to_fit_max_seq_len \
        --torch_dtype=float16 \
        --no-verbose

        # Clean unnecessary files
        clean_train_output_except_adapter(output_dir)

        # Logging + print update
        logging.info(f'Finished fine-tuning for split {idx}/{len(dataset_filepaths)}')
        print(f"âœ… Finished fine-tuning for split {idx}/{len(dataset_filepaths)} ({task_name})")

print("ğŸ�‰ All fine-tuning runs completed successfully!")


!ls -lh {checkpoints_folder}/*/checkpoint*/adapter_model.safetensors


# ------------------------------
# Wait for 2020 Solution Process
# ------------------------------
if not is_dry_run and cfg.ensemble_with_2020:
    # English: Old code to call program_search_dsl sequentially (commented out)
    # Ø§Ø±Ø¯Ùˆ: Ù¾Ø±Ø§Ù†Ø§ Ú©ÙˆÚˆ Ø¬Ùˆ program_search_dsl Ú©Ùˆ sequentially Ú©Ø§Ù„ Ú©Ø±ØªØ§ ØªÚ¾Ø§ (Ø§Ø¨ disable Û�Û’)
    #!python /kaggle/input/arc24-source-code/program_search_dsl.py \
    #--dataset_filepath={cfg.dataset_path} \
    #--output_filepath=submission_program_search.json
    
    # English: Notify that the system is waiting for the 2020 solution to finish
    # Ø§Ø±Ø¯Ùˆ: Ø§Ø·Ù„Ø§Ø¹ Ø¯ÛŒÙ†Ø§ Ú©Û� Ø³Ø³Ù¹Ù… 2020 Ø­Ù„ Ú©Û’ Ù…Ú©Ù…Ù„ Û�ÙˆÙ†Û’ Ú©Ø§ Ø§Ù†ØªØ¸Ø§Ø± Ú©Ø± Ø±Û�Ø§ Û�Û’
    print("â�³ Waiting for icecuber (2020 solution) process to finish...")

    # English: Wait for the process to end
    # Ø§Ø±Ø¯Ùˆ: Ù¾Ø±Ø§Ø³ÛŒØ³ Ú©Û’ Ø®ØªÙ… Û�ÙˆÙ†Û’ Ú©Ø§ Ø§Ù†ØªØ¸Ø§Ø± Ú©Ø±ÛŒÚº
    full_2020_solution_process.wait()

    # English: Capture both stdout and stderr after completion
    # Ø§Ø±Ø¯Ùˆ: Ù¾Ø±Ø§Ø³ÛŒØ³ Ú©Û’ Ù…Ú©Ù…Ù„ Û�ÙˆÙ†Û’ Ú©Û’ Ø¨Ø¹Ø¯ Ø¢Ø¤Ù¹ Ù¾Ù¹ Ø§ÙˆØ± Ø§ÛŒØ±Ø±Ø² Ø­Ø§ØµÙ„ Ú©Ø±ÛŒÚº
    stdout, stderr = full_2020_solution_process.communicate()

    print("ğŸ“¤ Script output:")
    print(stdout.decode())

    if stderr:
        print("âš ï¸� Script errors:")
        print(stderr.decode())
    else:
        print("âœ… No errors detected in script execution.")


%%time
if is_dry_run:
    with open('submission.json', 'w') as f:
        json.dump(dict(dry_run=True), f)
else:
    inference_path = 'inference'
    os.makedirs(inference_path)
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    dataset_filepaths = sorted(glob.glob(os.path.join(single_task_datasets_path, '*.json')))
    for dataset_filepath in tqdm(dataset_filepaths):
        task_id = os.path.splitext(os.path.basename(dataset_filepath))[0]
        checkpoint_path = os.path.join(checkpoints_folder, task_id, f'checkpoint-{cfg.max_steps}')
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint path does not exist: {checkpoint_path}')
            checkpoint_path = cfg.input_lora_path
        
        !python /kaggle/input/arc24-source-code/merge_lora.py \
        --base_model_path={cfg.model_path} \
        --lora_path={checkpoint_path} \
        --output_path={cfg.merged_model_path}
        
        output_filepath = os.path.join(inference_path, f'{task_id}_inference.json')
        while not os.path.exists(output_filepath):
            ! timeout {cfg.inference_timeout} python /kaggle/input/arc24-source-code/inference.py\
            --model_path={cfg.merged_model_path} \
            --prompt_version={cfg.prompt_version} \
            --dataset={dataset_filepath} \
            --output_filepath={output_filepath} \
            --max_model_len={cfg.max_model_len} \
            --grid_encoder="{cfg.grid_encoder}" \
            --predictions_per_task={cfg.predictions_per_task}  
            if not os.path.exists(output_filepath):
                print('\t\tWARNING, INFERENCE DID TIMEOUT!')
                
        logging.info(f'Finished inference for split {dataset_filepaths.index(dataset_filepath) + 1}/{len(dataset_filepaths)}')

# combine all the predictions into single files
if not is_dry_run:
    filepaths = glob.glob(os.path.join(inference_path, '*_inference.json'))
    solutions = dict()
    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            solutions.update(json.load(f))
    with open('submission_all.json', 'w') as f:
        json.dump(solutions, f)

    filepaths = glob.glob(os.path.join(inference_path, '*_task_results.json'))
    task_results = []
    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            task_results.extend(json.load(f))
    with open('submission_all_task_results.json', 'w') as f:
        json.dump(task_results, f)

if not is_dry_run:
    !python /kaggle/input/arc24-source-code/voting.py \
    --input_filepath=submission_all_task_results.json \
    --output_filepath=submission_voting.json  

if not is_dry_run and cfg.dataset_path != '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json':
    sys.path.append('/kaggle/input/arc25-source-code')
    from evaluation import (
        load_arc_data_with_solutions, evaluate,
        study_effect_of_the_number_of_solutions,
        study_attempt_accuracy,
        visualize_tasks_and_predictions)
    
    print('Results with all the predictions')
    with open('submission_all.json', 'r') as f:
        solutions = json.load(f)
    data = load_arc_data_with_solutions(cfg.dataset_path)
    evaluate(data, solutions)
    
    study_effect_of_the_number_of_solutions(solutions, data)
    visualize_tasks_and_predictions(solutions, data, only_correct=False)
    
    print('Results from selected 2 attemps')
    with open('submission_voting.json', 'r') as f:
        solutions = json.load(f)
    evaluate(data, solutions)
    study_attempt_accuracy(solutions, data)


if not is_dry_run:
    if cfg.ensemble_with_2020:
        !python /kaggle/input/arc24-source-code/combine_submissions.py \
        --sub_1=submission_program_search.json \
        --sub_2=icecuber_submission.json \
        --output=submission_2020.json \
        --give_preference_to_second_submission_on_second_attempt

        !python /kaggle/input/arc24-source-code/combine_submissions.py \
        --sub_1=submission_2020.json \
        --sub_2=submission_voting.json \
        --output=submission.json \
        --give_preference_to_second_submission_on_second_attempt
    else:
        !cp submission_voting.json submission.json

if not is_dry_run and cfg.dataset_path != '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json':
    print('Results from final submission')
    with open('submission.json', 'r') as f:
        solutions = json.load(f)
    evaluate(data, solutions)
    study_attempt_accuracy(solutions, data)


# ------------------------------
# Clean Workspace
# ------------------------------
def clean():
    """
    English: This function cleans the working directory by removing all files and folders 
             except 'submission.json'. It also clears temporary files from /kaggle/tmp.
    Ø§Ø±Ø¯Ùˆ: ÛŒÛ� Ù�Ù†Ú©Ø´Ù† ÙˆØ±Ú©Ù†Ú¯ ÚˆØ§Ø¦Ø±ÛŒÚ©Ù¹Ø±ÛŒ Ø³Û’ ØªÙ…Ø§Ù… Ù�Ø§Ø¦Ù„Ø² Ø§ÙˆØ± Ù�ÙˆÙ„ÚˆØ±Ø² Ú©Ùˆ ÚˆÛŒÙ„ÛŒÙ¹ Ú©Ø± Ø¯ÛŒØªØ§ Û�Û’
          Ø³ÙˆØ§Ø¦Û’ 'submission.json' Ú©Û’Û” Ø§Ø³ Ú©Û’ Ø¹Ù„Ø§ÙˆÛ� /kaggle/tmp Ø³Û’ Ø¨Ú¾ÛŒ Ø¹Ø§Ø±Ø¶ÛŒ Ù�Ø§Ø¦Ù„Ø² Û�Ù¹Ø§ØªØ§ Û�Û’Û”
    """
    print("ğŸ§¹ Cleaning workspace...")

    for filepath in glob.glob('*'):
        if filepath == 'submission.json':
            # English: Skip submission.json (we still need it)
            # Ø§Ø±Ø¯Ùˆ: submission.json Ú©Ùˆ Ù†Û� Û�Ù¹Ø§Ø¦ÛŒÚº (ÛŒÛ� Ø§Ø¨Ú¾ÛŒ Ú†Ø§Û�ÛŒÛ’)
            print(f"â�­ï¸� Skipping: {filepath}")
            continue

        if os.path.isdir(filepath):
            # English: Remove directory
            # Ø§Ø±Ø¯Ùˆ: Ù�ÙˆÙ„ÚˆØ± Ú©Ùˆ Û�Ù¹Ø§ Ø¯ÛŒÚº
            shutil.rmtree(filepath)
            print(f"ğŸ“‚ Deleted directory: {filepath}")
        else:
            # English: Remove file
            # Ø§Ø±Ø¯Ùˆ: Ù�Ø§Ø¦Ù„ Ú©Ùˆ Û�Ù¹Ø§ Ø¯ÛŒÚº
            os.remove(filepath)
            print(f"ğŸ“„ Deleted file: {filepath}")

    # English: Also clean Kaggle temp folder
    # Ø§Ø±Ø¯Ùˆ: Kaggle Ú©Ø§ Ø¹Ø§Ø±Ø¶ÛŒ Ù�ÙˆÙ„ÚˆØ± Ø¨Ú¾ÛŒ ØµØ§Ù� Ú©Ø±ÛŒÚº
    !rm -rf /kaggle/tmp/*
    print("ğŸ—‘ï¸� Temporary folder /kaggle/tmp cleared.")

    print("âœ… Workspace cleanup complete.")


# Run cleanup
clean()

# English: Show remaining files with details
# Ø§Ø±Ø¯Ùˆ: Ø¨Ø§Ù‚ÛŒ Ù�Ø§Ø¦Ù„Ø² Ú©Ùˆ ØªÙ�ØµÛŒÙ„ Ú©Û’ Ø³Ø§ØªÚ¾ Ø¯Ú©Ú¾Ø§Ø¦ÛŒÚº
!ls -lh


# # ------------------------------
# # Stop & Plot Resource Monitor
# # ------------------------------
# if not is_dry_run:
#     # English: Stop the resource monitor
#     # Ø§Ø±Ø¯Ùˆ: Ø±ÛŒØ³ÙˆØ±Ø³ Ù…Ø§Ù†ÛŒÙ¹Ø± Ú©Ùˆ Ø¨Ù†Ø¯ Ú©Ø±ÛŒÚº
#     print("â�¹ï¸� Stopping resource monitor...")
#     monitor.stop()
#     print("âœ… Monitor stopped.")

#     # English: Plot resource usage (CPU, memory, etc.)
#     # Ø§Ø±Ø¯Ùˆ: Ø±ÛŒØ³ÙˆØ±Ø³ Ø§Ø³ØªØ¹Ù…Ø§Ù„ (CPUØŒ Ù…ÛŒÙ…ÙˆØ±ÛŒ ÙˆØºÛŒØ±Û�) Ú©Ø§ Ú¯Ø±Ø§Ù� Ø¯Ú©Ú¾Ø§Ø¦ÛŒÚº
#     print("ğŸ“Š Plotting resource usage...")
#     monitor.plot()
#     print("âœ… Resource usage plot generated.")
# else:
#     print("ğŸ’¡ Dry run mode: Skipping monitor stop and plot.")

if not is_dry_run:
    monitor.stop()
    monitor.plot()

