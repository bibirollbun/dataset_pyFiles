!pip install transformers==4.49.0


%%capture
# Install necessary packages for fast model inference and GPU management.
!pip install unsloth vllm==0.7.3
!pip install triton==3.1.0  
!pip install -U pynvml


from unsloth import FastLanguageModel, PatchFastRL

# Patch the FastLanguageModel to integrate GRPO-specific modifications.
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch

# Set maximum sequence length and LoRA rank (controls the adaptation complexity).
max_seq_length = 2048  # Increase if you need longer reasoning traces.
lora_rank = 64         # Larger rank can improve performance but may slow down training.

# Load the Qwen model in 4-bit mode for reduced memory usage and enable fast inference with vLLM.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,           # Set to False if using LoRA in 16-bit precision.
    fast_inference = True,         # Enable vLLM for faster inference.
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,  # Adjust GPU memory usage to avoid out-of-memory errors.
)

# Wrap the model with PEFT (Parameter-Efficient Fine-Tuning) using LoRA.
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,           # Use a rank greater than 0; common choices include 8, 16, 32, 64, or 128.
    lora_alpha = lora_rank,  # A higher lora_alpha value means that the LoRA layers have a greater influence on the model's output, 
                             # while a lower value reduces this influence
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj",
    ],                                       # Specify target modules; you can remove QKVO if memory is limited.
    use_gradient_checkpointing = "unsloth",  # Enable gradient checkpointing for long context finetuning.
    random_state = 2025,                     # Set a random seed for reproducibility.
)


import re
from datasets import load_dataset, Dataset

# Define a system prompt that specifies the desired response format.
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# XML-based chain-of-thought (CoT) format template.
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_hash_answer(text: str) -> str | None:
    # Extracts the answer text that follows the "####" marker.
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Function to load and prepare the GSM8K dataset.
def get_gsm8k_questions(split = "train") -> Dataset:
    # Load the dataset from the 'openai/gsm8k' repository.
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    # Map the dataset to include a formatted prompt and the extracted answer.
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore

# Create the dataset to be used for training.
dataset = get_gsm8k_questions()


def extract_xml_answer(text: str) -> str:
    # Extracts the answer portion from the XML formatted text.
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward function to compare correctness of the generated answer.
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", 
          f"\nExtracted:\n{extracted_responses[0]}")
    # Award a reward if the extracted response matches the expected answer.
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# Reward function for integer responses.
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # Provide a reward if the response is a digit.
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# Reward function checking for strict XML format.
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion strictly adheres to the XML format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    # Reward if the response exactly matches the pattern.
    return [0.5 if match else 0.0 for match in matches]

# Reward function checking for a soft XML format (less strict).
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion follows the expected XML format (soft check)."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Helper function to count XML tags and provide a reward score based on counts.
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

# Reward function using XML tag counts to evaluate formatting.
def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


from trl import GRPOConfig, GRPOTrainer

# Configure GRPO training parameters.
# This configuration sets up the training hyperparameters, optimization settings, and inference acceleration via vLLM.
training_args = GRPOConfig(
    # use_vllm = True,                     # Enable vLLM to accelerate inference during training.
    learning_rate = 5e-6,                # Set the learning rate for the optimizer.
    adam_beta1 = 0.9,                    # First beta parameter for the AdamW optimizer.
    adam_beta2 = 0.99,                   # Second beta parameter for the AdamW optimizer.
    weight_decay = 0.1,                  # Weight decay to regularize the model and prevent overfitting.
    warmup_ratio = 0.1,                  # Fraction of steps used for learning rate warmup.
    lr_scheduler_type = "cosine",        # Use cosine annealing for the learning rate scheduler.
    optim = "adamw_8bit",                # Use 8-bit AdamW optimizer for memory efficiency.
    logging_steps = 1,                   # Log training information every step.
    bf16 = is_bfloat16_supported(),      # Use bfloat16 precision if supported by the GPU.
    fp16 = not is_bfloat16_supported(),  # Otherwise, fall back to fp16 precision.
    per_device_train_batch_size = 1,     # Batch size per device during training.
    gradient_accumulation_steps = 1,     # Accumulate gradients over this many steps (increase for smoother training if needed).
    num_generations = 8,                 # Number of generations per prompt (reduce if memory issues occur).
    max_prompt_length = 256,             # Maximum length for the input prompt.
    max_completion_length = 200,         # Maximum length for the generated completion.
    # num_train_epochs = 1,               # Uncomment this line to run training for one epoch.
    max_steps = 250,                     # Maximum number of training steps.
    save_steps = 250,                    # Save the model checkpoint every specified number of steps.
    max_grad_norm = 0.1,                 # Maximum gradient norm for gradient clipping.
    report_to = "none",                  # Disable reporting to external services like WandB.
    output_dir = "outputs",              # Directory to save the training outputs and checkpoints.
)

# Instantiate the GRPO trainer with the model, tokenizer, reward functions, and training dataset.
trainer = GRPOTrainer(
    model = model,                       # The language model to be trained.
    processing_class = tokenizer,        # The tokenizer used to preprocess the data.
    reward_funcs = [
        xmlcount_reward_func,            # Reward function based on XML tag counts.
        soft_format_reward_func,         # Reward function checking for soft adherence to XML formatting.
        strict_format_reward_func,       # Reward function checking for strict XML formatting.
        int_reward_func,                 # Reward function that provides rewards for integer outputs.
        correctness_reward_func,         # Reward function evaluating the correctness of the answer.
    ],
    args = training_args,                # GRPO training configuration.
    train_dataset = dataset,             # The training dataset containing prompts and expected answers.
)

# Begin training using the GRPO algorithm.
trainer.train()

# Save the LoRA-adapted model for later use.
model.save_lora("grpo_saved_lora")

