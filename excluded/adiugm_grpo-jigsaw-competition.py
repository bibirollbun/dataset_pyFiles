%%capture
!pip install unsloth


from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import re
import pandas as pd
from typing import Optional
from datasets import Dataset
from sklearn.model_selection import train_test_split


USER_PROMPT: str = """
<subreddit>{subreddit}</subreddit>
<rule>{rule}</rule>
<violation-example1>{violation_example1}</violation-example1>
<violation-example2>{violation_example2}</violation-example2>
<no-violation-example1>{no_violation_example1}</no-violation-example1>
<no-violation-example2>{no_violation_example2}</no-violation-example2>
<comment>{comment}</comment>
"""


SYSTEM_PROMPT: str = """
Here’s a refined and improved version of your prompt, with clearer structure, more precise instructions, and better examples to reduce ambiguity and improve consistency in responses:

Prompt:
You are an AI assistant specialized in evaluating Reddit comments for rule violations. Your task is to analyze each comment and assign a decimal score between -1 and 1, where:

-1.0: Extremely likely to violate the rule (clear, unambiguous violation).
1.0: Extremely unlikely to violate the rule (clearly compliant).
Avoid scores between -0.2 and 0.2—these are considered ambiguous and should not be used.

Input Format:
You will receive the following for each evaluation:

<subreddit>: The subreddit name.
<rule>: The specific rule to evaluate.
<violation-example1> and <violation-example2>: Two examples of comments that violate the rule.
<no-violation-example1> and <no-violation-example2>: Two examples of comments that do not violate the rule.
<comment>: The comment to evaluate.

Output Format:
Your response must be formatted as follows:

<label>: A decimal value between -1 and 1, where closer to -1 means more likely a violation and closer to 1 means more likely not a violation.
<think>: Your reasoning which should be concise and no more than 1 paragraph.

Do not include any other tags or text in your response.

Example Input:
<subreddit>funny</subreddit>
<rule>No greetings allowed</rule>
<violation-example1>Hello, my name is XYZ</violation-example1>
<violation-example2>Hi all! How are you?</violation-example2>
<no-violation-example1>My name is ABC</no-violation-example1>
<no-violation-example2>Last week I saw a bird</no-violation-example2>
<comment>Hi, I was walking the other day and saw something really funny</comment>

Example Output:
<label>-0.9</label>
<think>The comment starts with "Hi," which is a greeting and matches the violation examples. The presence of a greeting is sufficient for a violation.</think>

Classify the following text as requested above.
"""


FORMATTING_PATTERNS: list[tuple[str, float]] = [
    (r'<think>.*?</think>', 0.5),
    (r'<label>.*?</label>', 0.5)
]


def check_str_format(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))

def extract_field_from_text(text: str, pattern: str, pos: int = 1) -> str:
    match: Optional[re.Match] = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    try:
        return match.group(pos).strip()

    except AttributeError:
        raise

def add_user_prompt_dataset(
    dataset: Dataset,
    subreddit_col: str = 'subreddit',
    rule_col: str = 'rule',
    no_violation_example1_col: str = 'negative_example_1',
    no_violation_example2_col: str = 'negative_example_2',
    violation_example1_col: str = 'positive_example_1',
    violation_example2_col: str = 'positive_example_2',
    comment_col: str = 'body'
):
    return dataset.map(lambda x: {
    'prompt': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT.format(
            subreddit = x[subreddit_col],
            rule = x[rule_col],
            violation_example1 = x[violation_example1_col],
            violation_example2 = x[violation_example2_col],
            no_violation_example1 = x[no_violation_example1_col],
            no_violation_example2 = x[no_violation_example2_col],                
            comment = x[comment_col],
        )},
    ],
    'rule_violation': 1 if x['rule_violation'] == 0 else -1
})


def calculate_reward_formatting(text: str, patterns: list[tuple[str, float]]) -> float:
    reward = 0.0
    for pattern, weight in patterns:
        reward += int(check_str_format(text=text, pattern=pattern)) * weight
    return reward

def calculate_reward_soft_binary_label(
    text: str,
    true_label: int,
    zero_interval: tuple[float, float] = (-0.2, 0.2),
    label_pattern: str = r"<label>(.*?)</label>"
    ) -> float:

    try:
        predicted_label = float(extract_field_from_text(text=text, pattern=label_pattern))

    except Exception:
        return 0

    if predicted_label >= zero_interval[0] and predicted_label <= zero_interval[1]:
        return 0

    try:
        if true_label == 1:
            to_return = (2 - (true_label-predicted_label))/2
        else:
            to_return = (2 - (predicted_label-true_label))/2
            
    except Exception:
        return 0
    else:
        return to_return


df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['rule_violation'],
    random_state=42
)

# Convert the DataFrames back to datasets.Dataset if needed
dataset_train: Dataset = Dataset.from_pandas(train_df)
dataset_test: Dataset = Dataset.from_pandas(test_df)


dataset_train = add_user_prompt_dataset(dataset_train)


def reward_formatting_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [
        calculate_reward_formatting(
            text=response,
            patterns=FORMATTING_PATTERNS
        ) for response in responses
    ]

def reward_soft_binary_func(completions, rule_violation, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [
        calculate_reward_soft_binary_label(
            text=response,
            true_label=rule_violation[0]     
        ) for response in responses
    ]


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length = 4096,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = False, # Enable vLLM fast inference - does not work
    max_lora_rank = 256,
    gpu_memory_utilization = 0.4, # Reduce if out of memory
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 256, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = 256 * 2,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 42,
)


training_args = GRPOConfig(
    use_vllm = False, # use vLLM for fast inference!
    learning_rate = 2e-5, #changed this because of LoRA
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    beta=0.01,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
   # bf16 = is_bfloat16_supported(),
   # fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 5000,
    max_completion_length = 400,
    #num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 3,
    save_steps = 3,
    max_grad_norm = 0.9,
    report_to = "none", # Can use Weights & Biases
    output_dir = "/kaggle/working/",
)


training_args.beta


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        reward_formatting_func,
        reward_soft_binary_func
    ],
    reward_weights = [0.3,0.7],
    args = training_args,
    train_dataset = dataset_train,
)


trainer.train()
df_results = pd.DataFrame.from_records(trainer.state.log_history)


model.save_pretrained("./kaggle/working/model_at_1_epoch")
tokenizer.save_pretrained("./kaggle/working/tokenizer_at_1_epoch")




%%writefile train_grpo.py
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import re
import pandas as pd
from typing import Optional
from datasets import Dataset
from sklearn.model_selection import train_test_split

USER_PROMPT: str = """
<subreddit>{subreddit}</subreddit>
<rule>{rule}</rule>
<violation-example1>{violation_example1}</violation-example1>
<violation-example2>{violation_example2}</violation-example2>
<no-violation-example1>{no_violation_example1}</no-violation-example1>
<no-violation-example2>{no_violation_example2}</no-violation-example2>
<comment>{comment}</comment>
"""

SYSTEM_PROMPT: str = """
Here’s a refined and improved version of your prompt, with clearer structure, more precise instructions, and better examples to reduce ambiguity and improve consistency in responses:

Prompt:
You are an AI assistant specialized in evaluating Reddit comments for rule violations. Your task is to analyze each comment and assign a decimal score between -1 and 1, where:

-1.0: Extremely likely to violate the rule (clear, unambiguous violation).
1.0: Extremely unlikely to violate the rule (clearly compliant).
Avoid scores between -0.2 and 0.2—these are considered ambiguous and should not be used.

Input Format:
You will receive the following for each evaluation:

<subreddit>: The subreddit name.
<rule>: The specific rule to evaluate.
<violation-example1> and <violation-example2>: Two examples of comments that violate the rule.
<no-violation-example1> and <no-violation-example2>: Two examples of comments that do not violate the rule.
<comment>: The comment to evaluate.

Output Format:
Your response must be formatted as follows:

<label>: A decimal value between -1 and 1, where closer to -1 means more likely a violation and closer to 1 means more likely not a violation.
<think>: Your reasoning which should be concise and no more than 1 paragraph.

Do not include any other tags or text in your response.

Example Input:
<subreddit>funny</subreddit>
<rule>No greetings allowed</rule>
<violation-example1>Hello, my name is XYZ</violation-example1>
<violation-example2>Hi all! How are you?</violation-example2>
<no-violation-example1>My name is ABC</no-violation-example1>
<no-violation-example2>Last week I saw a bird</no-violation-example2>
<comment>Hi, I was walking the other day and saw something really funny</comment>

Example Output:
<label>-0.9</label>
<think>The comment starts with "Hi," which is a greeting and matches the violation examples. The presence of a greeting is sufficient for a violation.</think>

Classify the following text as requested above.
"""

FORMATTING_PATTERNS: list[tuple[str, float]] = [
    (r'<think>.*?</think>', 0.5),
    (r'<label>.*?</label>', 0.5)
]

def check_str_format(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL))

def extract_field_from_text(text: str, pattern: str, pos: int = 1) -> str:
    match: Optional[re.Match] = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    try:
        return match.group(pos).strip()

    except AttributeError:
        raise

def add_user_prompt_dataset(
    dataset: Dataset,
    subreddit_col: str = 'subreddit',
    rule_col: str = 'rule',
    no_violation_example1_col: str = 'negative_example_1',
    no_violation_example2_col: str = 'negative_example_2',
    violation_example1_col: str = 'positive_example_1',
    violation_example2_col: str = 'positive_example_2',
    comment_col: str = 'body'
):
    return dataset.map(lambda x: {
    'prompt': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_PROMPT.format(
            subreddit = x[subreddit_col],
            rule = x[rule_col],
            violation_example1 = x[violation_example1_col],
            violation_example2 = x[violation_example2_col],
            no_violation_example1 = x[no_violation_example1_col],
            no_violation_example2 = x[no_violation_example2_col],                
            comment = x[comment_col],
        )},
    ],
    'rule_violation': 1 if x['rule_violation'] == 0 else -1
})

def calculate_reward_formatting(text: str, patterns: list[tuple[str, float]]) -> float:
    reward = 0.0
    for pattern, weight in patterns:
        reward += int(check_str_format(text=text, pattern=pattern)) * weight
    return reward

def calculate_reward_soft_binary_label(
    text: str,
    true_label: int,
    zero_interval: tuple[float, float] = (-0.2, 0.2),
    label_pattern: str = r"<label>(.*?)</label>"
    ) -> float:

    try:
        predicted_label = float(extract_field_from_text(text=text, pattern=label_pattern))

    except Exception:
        return 0

    if predicted_label >= zero_interval[0] and predicted_label <= zero_interval[1]:
        return 0

    try:
        if true_label == 1:
            to_return = (2 - (true_label-predicted_label))/2
        else:
            to_return = (2 - (predicted_label-true_label))/2
            
    except Exception:
        return 0
    else:
        return to_return

def reward_formatting_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [
        calculate_reward_formatting(
            text=response,
            patterns=FORMATTING_PATTERNS
        ) for response in responses
    ]

def reward_soft_binary_func(completions, rule_violation, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    return [
        calculate_reward_soft_binary_label(
            text=response,
            true_label=rule_violation[0]     
        ) for response in responses
    ]

def main():

    df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['rule_violation'],
        random_state=42
    )
    
    # Convert the DataFrames back to datasets.Dataset if needed
    dataset_train: Dataset = Dataset.from_pandas(train_df)
    dataset_test: Dataset = Dataset.from_pandas(test_df)
    
    dataset_train = add_user_prompt_dataset(dataset_train)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length = 4096,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = False, # Enable vLLM fast inference - does not work
        max_lora_rank = 256,
        gpu_memory_utilization = 0.4, # Reduce if out of memory
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = 256, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = 256 * 2,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 42,
    )

    training_args = GRPOConfig(
        use_vllm = False, # use vLLM for fast inference!
        learning_rate = 2e-5, #changed this because of LoRA
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        beta=0.01,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
       # bf16 = is_bfloat16_supported(),
       # fp16 = not is_bfloat16_supported(),
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 8, # Decrease if out of memory
        max_prompt_length = 5000,
        max_completion_length = 400,
        #num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 3,
        save_steps = 3,
        max_grad_norm = 0.9,
        report_to = "none", # Can use Weights & Biases
        output_dir = "/kaggle/working/",
    )
    
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_formatting_func,
            reward_soft_binary_func
        ],
        reward_weights = [0.3,0.7],
        args = training_args,
        train_dataset = dataset_train,
    )
    
    trainer.train()
    df_results = pd.DataFrame.from_records(trainer.state.log_history)

    #model.save_pretrained("/kaggle/working/model_at_1_epoch")
    #tokenizer.save_pretrained("/kaggle/working/tokenizer_at_1_epoch")

    model.save_pretrained("/kaggle/working/model_test")
    tokenizer.save_pretrained("/kaggle/working/tokenizer_test")

if __name__ == "__main__":
    main()


from accelerate import Accelerator
accelerator = Accelerator()


!accelerate launch train_grpo.py




