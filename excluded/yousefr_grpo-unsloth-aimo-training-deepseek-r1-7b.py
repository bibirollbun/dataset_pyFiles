from IPython.display import clear_output
!python -m pip install --no-index -v --find-links=/kaggle/input/unsloth-script unsloth --pre
!python -m pip install --no-index -v --find-links=/kaggle/input/aimo-packages/offline_packages vllm --pre

clear_output()


from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)


import os
import re
import pandas as pd
from datasets import Dataset
from accelerate import Accelerator

from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
import torch

import re
import torch
import numpy as np
import pandas as pd

from vllm import SamplingParams

import transformers
import statistics

seed = 2025

transformers.set_seed(seed)
torch.manual_seed(seed)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "true"



def extract_last_integer(s: str, dummy_value = None):
    """
    Extracts the last integer from the given string, prioritizing numbers in \boxed{}.
    If no number is found, returns the dummy value.

    :param s: Input string.
    :param dummy_value: Value to return if no number is found.
    :return: The last integer found, preferring boxed numbers.
    """
    # Look for numbers inside \boxed{}
    boxed_numbers = re.findall(r"\\boxed\{(-?\d+)\}", s) # return number whether positive or negative
    if boxed_numbers:
        return int(boxed_numbers[-1])  # Return the last boxed number found
    
    # Find all standalone integers (including negatives)
    nums = re.findall(r'-?\d+', s)  
    if nums:
        return int(nums[-1])  # Return the last number found, considering negative sign
    
    return dummy_value  # Return dummy value if no number found

def filter_dataframe_by_solution(df: pd.DataFrame, solution_column: str):
    """
    Removes all rows that don't have numbers in the specified solution column.

    :param df: Input DataFrame.
    :param solution_column: Column name to check for numbers.
    :return: Filtered DataFrame.
    """
    df['answer'] = df[solution_column].astype(str).apply(extract_last_integer)
    # filter None from df['answer'] before returning
    return df[df['answer'].notna()].reset_index(drop=True)

def filter_dataframe_by_string_length(df: pd.DataFrame, max_length: int):
    """
    Removes all rows that have any column with a string length below the given threshold.

    :param df: Input DataFrame.
    :param min_length: Minimum string length required for all columns.
    :return: Filtered DataFrame.
    """
    return df[df.apply(lambda row: all(len(str(value)) <= max_length for value in row), axis=1)].reset_index(drop=True)


dataset = pd.read_parquet('/kaggle/input/math-problems-imo/math_problems.parquet')

context_length = 1000

dataset = filter_dataframe_by_string_length(dataset, context_length)
dataset = filter_dataframe_by_solution(dataset, 'solution')

dataset = dataset[:100]


lora_rank = 16 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length = context_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
    enforce_eager = True,
    float8_kv_cache = True,
    dtype = "float16"
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


def create_prompt(sample, inference = False):
    question = sample['problem']
    if not inference: msg = "The user asks a question, and the Assistant solves it.  The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>"
    else: msg = "A conversation between User and Assistant. The user asks a question, and the Assistant who is a math expert solves it. The assisstant always answers correctly. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
    chat = [{"role": "system", "content": msg},
            {"role": "user", "content": question + ' Return final answer within \\boxed{}, after taking modulo 1000.'},]
    sample['prompt'] = tokenizer.apply_chat_template(
            conversation=chat,
            tokenize=False,
            add_generation_prompt=True
        )
    return sample


dataset = Dataset.from_pandas(dataset)
dataset = dataset.map(create_prompt)

# Filter dataset after adding the prompt format

def filter_by_question_length(example):
    """Removes rows where the 'question' column length exceeds context_length."""
    return len(example["prompt"]) <= context_length
    
dataset = dataset.filter(filter_by_question_length)
dataset = dataset.train_test_split(test_size=0.1)

dataset


def gen(model, text_list, max_tokens, decode):
    """
    Generates batched outputs for a list of texts.

    :param model: The model used for text generation.
    :param text_list: List of input texts.
    :param max_tokens: Maximum number of tokens to generate.
    :return: List of generated outputs.
    """

    sampling_params = SamplingParams(
        temperature=0.2,
        min_p=0.01,
        max_tokens=max_tokens,
        n=1,  # Generate 1 response per question
    )
    
    # Run inference
    results = model.fast_generate(text_list, sampling_params)

    if decode: output_texts = [result.prompt for result in results]
    else: output_texts = [result.outputs[0].token_ids for result in results]

    return output_texts

def evaluate_rewards(model, dataset, reward_functions: dict[str, callable], max_tokens: int = 4024, decode: bool = False):
    
    FastLanguageModel.for_inference(model)
    completions = []

    formatted_prompts = [create_prompt({"problem": q}, inference = True)['prompt'] for q in dataset["problem"]]
    completion = gen(model, formatted_prompts, max_tokens, decode)
    completions.extend(completion)

    res = {}
    for nm, reward_func in enumerate(reward_functions):
        
        try: v = reward_func(completions = completions)
        except Exception as e: print(f"Couldn't process reward function: {e}")
            
        print(nm, np.mean(v))
        res[nm] = np.mean(v)

    model.train()

    return res


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = completions
    q = prompts[0]
    extracted_responses = [extract_last_integer(r) for r in responses]
    return sum([10.0 if r == answer[0] else -1.0 for r in extracted_responses]) / len(responses)

def check_boxed_value(s: str):
    
    match = re.search(r"\\boxed\{([^}]*)\}", s)  # Extract content inside \boxed{}
    
    if match:
        content = match.group(1).strip()
        
        if content: return 0
    
    return -0.3  # Return -0.3 if no \boxed{} is found

def check_boxed_values_for_responses(completions, **kwargs) -> list[float]:
    return [check_boxed_value(c) for c in completions]


reward_funcs = [
        correctness_reward_func,
        check_boxed_values_for_responses
    ]


evaluate_rewards(model, dataset["test"], reward_funcs, decode = True)


training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 1e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 5,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 3, # Decrease if out of memory
    max_prompt_length = context_length,
    max_completion_length = context_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
    seed = 2025,
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = reward_funcs,
    args = training_args,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
)



trainer.train()


evaluate_rewards(model, dataset["test"], reward_funcs, max_tokens = 4049)


model.save_pretrained_gguf("model", tokenizer, quantization_method = "quantized")


lora_rank = 32
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "model",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
    enforce_eager = True,
    use_gradient_checkpointing = False
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


reference_csv_path = "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
reference_df = pd.read_csv(reference_csv_path)

def select_answer(answers):
    """Selects the most frequent valid integer answer, modulo 1000."""
    answer = statistics.mode(answers)
    return answer % 1000

# Define prompt types
prompt_types = ['basic']  # Expand as needed
accuracy_results = {ptype: 0 for ptype in prompt_types}

# Generate formatted prompts
formatted_prompts = [create_prompt({"problem": q}, inference = True)['prompt'] for q in reference_df["problem"].tolist()]

results = gen(model, formatted_prompts, max_seq_length, decode = False)

# Compare generated answers with reference answers
correct_count = 0
total = len(reference_df)

for i, output in enumerate(results):
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True)
    ]

    print(f"\nğŸ”¹ Generated outputs for: {reference_df.iloc[i]['problem']}")
    print(generated_texts)

    raw_answers = [extract_last_integer(text) for text in generated_texts]
    filtered_answers = [ans for ans in raw_answers if ans is not None]
    best_answer = select_answer(filtered_answers)

    if str(best_answer) == str(reference_df.iloc[i]["answer"]):
        correct_count += 1

# Store accuracy results
accuracy_results['basic'] = correct_count / total
accuracy_results_df = pd.DataFrame(list(accuracy_results.items()), columns=["Prompt Type", "Accuracy"])
accuracy_results_df.to_csv("prompt_accuracy_results.csv", index=False)

print("\nğŸ�¯ **Prompt Engineering Accuracy Results**")
print(accuracy_results_df)
print("\nğŸ“� Results saved to 'prompt_accuracy_results.csv'")



df = pd.read_csv("/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv")
df




