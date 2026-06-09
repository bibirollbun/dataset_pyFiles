


!pip install unsloth vllm
!pip install --upgrade pillow


from unsloth import FastLanguageModel
import torch

max_seq_length=1024
lora_rank = 32


model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6,
)


model  = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",

    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407
)



from trl import GRPOConfig,GRPOTrainer
max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Increase to 4 for smoother training
    num_generations=2,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=256,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
)


import pandas as pd
import re
from datasets import Dataset
from typing import List, Dict

# 1. Define the System Prompt for the Model (No change)
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

# 2. Helper function to create the detailed user prompt (No change)
def create_user_prompt(row: pd.Series) -> str:
    """Creates a detailed prompt for the user part of the conversation."""
    return f"""Analyze the following content for rule violations:

POST: "{row['body']}"

RULE: "{row['rule']}"

POSITIVE EXAMPLES (Allowed Content):
1. "{row['positive_example_1']}"
2. "{row['positive_example_2']}"

NEGATIVE EXAMPLES (Rule Violations):
1. "{row['negative_example_1']}"
2. "{row['negative_example_2']}"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# 3. Dataset Preparation Function (No change)
def get_content_moderation_dataset(df: pd.DataFrame) -> Dataset:
    """
    Transforms the content moderation DataFrame into a Hugging Face Dataset
    formatted for the GRPOTrainer.
    """
    data_list = []
    for _, row in df.iterrows():
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": create_user_prompt(row)},
        ]
        answer = {
            "rule_violation": int(row["rule_violation"]),
            "rule": row["rule"]
        }
        data_list.append({"prompt": prompt, "answer": answer})
    return Dataset.from_list(data_list)

# 4. Helper function to parse model responses (No change)
def parse_completion(text: str) -> Dict[str, str]:
    """Extracts structured fields from the model's text completion."""
    classification = re.search(r"CLASSIFICATION:\s*(VIOLATION|NO_VIOLATION)", text, re.IGNORECASE)
    confidence = re.search(r"CONFIDENCE:\s*([0-9\.]+)", text, re.IGNORECASE)
    reasoning = re.search(r"REASONING:\s*(.*?)(?=\n\w+:|$)", text, re.IGNORECASE | re.DOTALL)
    return {
        "classification": classification.group(1).strip() if classification else "N/A",
        "confidence": confidence.group(1).strip() if confidence else "0.0",
        "reasoning": reasoning.group(1).strip() if reasoning else ""
    }

# 5. Define the Reward Functions (CORRECTED)
# The signature is now `(**kwargs)` and we get `answers` from `kwargs['answer']`.

def correctness_reward_func(prompts: list, completions: list, **kwargs) -> list[float]:
    """Rewards the model for correct classification (VIOLATION vs. NO_VIOLATION)."""
    answers = kwargs["answer"] # ✅ Get answers from kwargs
    rewards = []
    responses = [comp[0]["content"] for comp in completions]
    
    for response, ground_truth in zip(responses, answers):
        parsed = parse_completion(response)
        correct_label = "VIOLATION" if ground_truth['rule_violation'] == 1 else "NO_VIOLATION"
        
        if parsed["classification"].upper() == correct_label:
            rewards.append(2.0)
        else:
            rewards.append(0.0)
            
    return rewards

def reasoning_quality_reward_func(prompts: list, completions: list, **kwargs) -> list[float]:
    """Rewards the model for providing detailed and relevant reasoning."""
    answers = kwargs["answer"] # ✅ Get answers from kwargs
    rewards = []
    responses = [comp[0]["content"] for comp in completions]
    quality_indicators = ["rule", "post", "content", "example", "community", "guideline"]

    for response in responses:
        parsed = parse_completion(response)
        reasoning_text = parsed["reasoning"].lower()
        length_score = min(len(reasoning_text.split()) / 40.0, 1.0)
        indicator_score = sum(1 for ind in quality_indicators if ind in reasoning_text) / len(quality_indicators)
        total_reward = 0.5 * (length_score + indicator_score)
        rewards.append(total_reward)
        
    return rewards

def format_compliance_reward_func(prompts: list, completions: list, **kwargs) -> list[float]:
    """Rewards the model for strictly following the required output format."""
    answers = kwargs["answer"] # ✅ Get answers from kwargs
    rewards = []
    responses = [comp[0]["content"] for comp in completions]
    required_sections = ["CLASSIFICATION:", "CONFIDENCE:", "REASONING:", "KEY_FACTORS:"]
    
    for response in responses:
        found_sections = sum(1 for section in required_sections if section in response)
        reward = (found_sections / len(required_sections)) * 0.5
        rewards.append(reward)
        
    return rewards
    



# --- Example Usage ---

# A. Create a sample DataFrame (replace with loading your actual CSV)
sample_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df = pd.DataFrame(sample_data)

# B. Prepare the dataset
dataset = get_content_moderation_dataset(df)

print("--- Sample Processed Dataset Entry ---")
print(dataset[0]['prompt'])
print("\n--- Ground Truth Answer ---")
print(dataset[0]['answer'])
print("-" * 35)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[
        correctness_reward_func,
        reasoning_quality_reward_func,
        format_compliance_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)


trainer.train()


# ---- IMPORTANT ----
# Immediately save the model after training is complete in the SAME cell.
print("Saving LoRA adapters...")
model.save_pretrained("grpo_saved_lora")
print("Model saved successfully!")


# Add this code in a new cell to test your model

from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
# We load the original model, which is not yet fine-tuned
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

# 2. Load the LoRA adapters from your saved folder
# This applies your fine-tuning to the base model
model = PeftModel.from_pretrained(model, "grpo_saved_lora")


# 3. Create the prompt using the same format as your training data
# This is CRUCIAL for getting good results.
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

# Let's create a new post to test
test_post = {
    'body': "Hey everyone, check out my new crypto project! Guaranteed 100x returns! Join my discord here: [link]",
    'rule': "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
    'positive_example_1': "Discussing the merits of a publicly traded company.",
    'positive_example_2': "Sharing a news article about a new technology.",
    'negative_example_1': "Buy my new ebook on how to get rich quick!",
    'negative_example_2': "PM me for details on this amazing investment opportunity.",
    'subreddit': "r/technology"
}

# This helper function is the same one from your training script
def create_user_prompt(row: dict) -> str:
    return f"""Analyze the following content for rule violations:

POST: \"{row['body']}\"

RULE: \"{row['rule']}\"

POSITIVE EXAMPLES (Allowed Content):
1. \"{row['positive_example_1']}\"
2. \"{row['positive_example_2']}\"

NEGATIVE EXAMPLES (Rule Violations):
1. \"{row['negative_example_1']}\"
2. \"{row['negative_example_2']}\"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# Combine the system and user prompts into a chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": create_user_prompt(test_post)},
]

# 4. Tokenize the input and generate a response
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs, **generation_kwargs)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("--- Model Analysis ---")
print(response)


# Add this code in a new cell to test your model

from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
# We load the original model, which is not yet fine-tuned
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

# 2. Load the LoRA adapters from your saved folder
# This applies your fine-tuning to the base model
model = PeftModel.from_pretrained(model, "grpo_saved_lora")


# 3. Create the prompt using the same format as your training data
# This is CRUCIAL for getting good results.
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

test_post = {
        'body': "Kickin' ass and selling underwear!\nJust made a new store, come check it out!",
        'rule': "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        'positive_example_1': "Good story my friend. Check out my blog at  http://noncommercialstory.com",
        'positive_example_2': "If you know what exactly you need then you don't need extra shopping tips.",
        'negative_example_1': "CENTIPEDES\n\nSOME BASED PATRIOTS HAVE CREATED...",
        'negative_example_2': "[So great! Thanks for sharing.](http://www.cheappromo.com)",
        'subreddit': "gonewild"
}


# This helper function is the same one from your training script
def create_user_prompt(row: dict) -> str:
    return f"""Analyze the following content for rule violations:

POST: \"{row['body']}\"

RULE: \"{row['rule']}\"

POSITIVE EXAMPLES (Allowed Content):
1. \"{row['positive_example_1']}\"
2. \"{row['positive_example_2']}\"

NEGATIVE EXAMPLES (Rule Violations):
1. \"{row['negative_example_1']}\"
2. \"{row['negative_example_2']}\"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# Combine the system and user prompts into a chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": create_user_prompt(test_post)},
]

# 4. Tokenize the input and generate a response
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs, **generation_kwargs)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("--- Model Analysis ---")
print(response)


# Add this code in a new cell to test your model

from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
# We load the original model, which is not yet fine-tuned
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

# 2. Load the LoRA adapters from your saved folder
# This applies your fine-tuning to the base model
model = PeftModel.from_pretrained(model, "grpo_saved_lora")


# 3. Create the prompt using the same format as your training data
# This is CRUCIAL for getting good results.
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

test_post = {
        'body': "watch  hooters  best  therein  http://clickandwatch.com",
        'rule': "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
        'positive_example_1': "Earn 50,000 bonus points with Chase Sapphire Preferred by meeting spending requirements.",
        'positive_example_2': "Cool, front page! I made this print along with the artist.",
        'negative_example_1': "[Full HD Movie Online Free](http://www.flickmania.com)",
        'negative_example_2': "* Karambit Black Pearl\n* 0.02137822 Float (unbox now)",
        'subreddit': "personalfinance"
}


# This helper function is the same one from your training script
def create_user_prompt(row: dict) -> str:
    return f"""Analyze the following content for rule violations:

POST: \"{row['body']}\"

RULE: \"{row['rule']}\"

POSITIVE EXAMPLES (Allowed Content):
1. \"{row['positive_example_1']}\"
2. \"{row['positive_example_2']}\"

NEGATIVE EXAMPLES (Rule Violations):
1. \"{row['negative_example_1']}\"
2. \"{row['negative_example_2']}\"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# Combine the system and user prompts into a chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": create_user_prompt(test_post)},
]

# 4. Tokenize the input and generate a response
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs, **generation_kwargs)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("--- Model Analysis ---")
print(response)


# Add this code in a new cell to test your model

from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
# We load the original model, which is not yet fine-tuned
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

# 2. Load the LoRA adapters from your saved folder
# This applies your fine-tuning to the base model
model = PeftModel.from_pretrained(model, "grpo_saved_lora")


# 3. Create the prompt using the same format as your training data
# This is CRUCIAL for getting good results.
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

test_post = {
    'body': "Forget the pharmacy! Get your pain relief meds like Oxy and Xanax without a prescription. Fast, discreet shipping. Visit us at BestMedsOnline . shop",
    'rule': "No illegal content or transactions: Do not encourage or facilitate any transaction or gift involving certain goods and services, including firearms, drugs, or stolen goods.",
    'positive_example_1': "Ibuprofen is a good over-the-counter option for mild pain.",
    'positive_example_2': "My doctor prescribed me medication after my surgery.",
    'negative_example_1': "Looking for advice on how to deal with chronic back pain.",
    'negative_example_2': "Has anyone had success with physical therapy?",
    'subreddit': "health"
}


# This helper function is the same one from your training script
def create_user_prompt(row: dict) -> str:
    return f"""Analyze the following content for rule violations:

POST: \"{row['body']}\"

RULE: \"{row['rule']}\"

POSITIVE EXAMPLES (Allowed Content):
1. \"{row['positive_example_1']}\"
2. \"{row['positive_example_2']}\"

NEGATIVE EXAMPLES (Rule Violations):
1. \"{row['negative_example_1']}\"
2. \"{row['negative_example_2']}\"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# Combine the system and user prompts into a chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": create_user_prompt(test_post)},
]

# 4. Tokenize the input and generate a response
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs, **generation_kwargs)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("--- Model Analysis ---")
print(response)


# Add this code in a new cell to test your model

from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
# We load the original model, which is not yet fine-tuned
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

# 2. Load the LoRA adapters from your saved folder
# This applies your fine-tuning to the base model
model = PeftModel.from_pretrained(model, "grpo_saved_lora")


# 3. Create the prompt using the same format as your training data
# This is CRUCIAL for getting good results.
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

test_post = {
    'body': "I'm working on a new project and trying to decide between using React or Vue. Does anyone have experience with both and can share some pros and cons?",
    'rule': "No Advertising: Spam, referral links, unsolicited advertising, and promotional content are not allowed.",
    'positive_example_1': "Check out my new SaaS product for developers!",
    'positive_example_2': "I built a tool that does exactly this, download it at mycooltool.com",
    'negative_example_1': "What are the best libraries for building UI components?",
    'negative_example_2': "I'm struggling with state management in my application.",
    'subreddit': "webdev"
}


# This helper function is the same one from your training script
def create_user_prompt(row: dict) -> str:
    return f"""Analyze the following content for rule violations:

POST: \"{row['body']}\"

RULE: \"{row['rule']}\"

POSITIVE EXAMPLES (Allowed Content):
1. \"{row['positive_example_1']}\"
2. \"{row['positive_example_2']}\"

NEGATIVE EXAMPLES (Rule Violations):
1. \"{row['negative_example_1']}\"
2. \"{row['negative_example_2']}\"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# Combine the system and user prompts into a chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": create_user_prompt(test_post)},
]

# 4. Tokenize the input and generate a response
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs, **generation_kwargs)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("--- Model Analysis ---")
print(response)


# Add this code in a new cell to test your model

from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# 1. Load the base model and tokenizer
# We load the original model, which is not yet fine-tuned
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    gpu_memory_utilization=0.6,
)

# 2. Load the LoRA adapters from your saved folder
# This applies your fine-tuning to the base model
model = PeftModel.from_pretrained(model, "grpo_saved_lora")


# 3. Create the prompt using the same format as your training data
# This is CRUCIAL for getting good results.
SYSTEM_PROMPT = """\
You are an expert content moderator. Analyze the user's post based on the provided rule and examples.
Respond in the following format, providing a detailed analysis.

CLASSIFICATION: [VIOLATION/NO_VIOLATION]
CONFIDENCE: [A float between 0.0 and 1.0]
REASONING: [Your detailed explanation for the classification]
KEY_FACTORS: [A list of the key factors that influenced your decision]
"""

test_post = {
    'body': "Honestly, anyone who still supports this politician is a complete moron. You'd have to be brain-dead to agree with their policies. It's just pathetic.",
    'rule': "Be civil. Personal attacks, insults, and hate speech are not tolerated.",
    'positive_example_1': "I disagree with that politician's stance on economic policy.",
    'positive_example_2': "Can someone explain the reasoning behind this new law? I don't understand the benefits.",
    'negative_example_1': "You are an idiot if you believe that.",
    'negative_example_2': "User u/SomeRedditor is a known liar and a shill.",
    'subreddit': "politics"
}


# This helper function is the same one from your training script
def create_user_prompt(row: dict) -> str:
    return f"""Analyze the following content for rule violations:

POST: \"{row['body']}\"

RULE: \"{row['rule']}\"

POSITIVE EXAMPLES (Allowed Content):
1. \"{row['positive_example_1']}\"
2. \"{row['positive_example_2']}\"

NEGATIVE EXAMPLES (Rule Violations):
1. \"{row['negative_example_1']}\"
2. \"{row['negative_example_2']}\"

SUBREDDIT: {row['subreddit']}

Based on this information, please provide your analysis in the required format.
"""

# Combine the system and user prompts into a chat format
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": create_user_prompt(test_post)},
]

# 4. Tokenize the input and generate a response
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

# Set generation parameters
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
}

# Generate the output
with torch.no_grad():
    outputs = model.generate(input_ids=inputs, **generation_kwargs)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

print("--- Model Analysis ---")
print(response)


# Save to 16-bit precision
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")


# Push to Hugging Face Hub (requires a token)
model.push_to_hub_merged(
    "PABITRA07/grpo-reddit-rule-model", tokenizer, save_method="merged_16bit", token="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"
)


# # Q4_K_M version
# model.push_to_hub_gguf(
#     "PABITRA07/grpo-reddit-rule-model-q4_k_m",
#     tokenizer,
#     quantization_method="q4_k_m",
#     token="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",
# )

# # Q8_0 version
# model.push_to_hub_gguf(
#     "PABITRA07/grpo-reddit-rule-model-q8_0",
#     tokenizer,
#     quantization_method="q8_0",
#     token="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",
# )

# # Q5_K_M version
# model.push_to_hub_gguf(
#     "PABITRA07/grpo-reddit-rule-model-q5_k_m",
#     tokenizer,
#     quantization_method="q5_k_m",
#     token="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",
# )



# import pandas as pd
# from vllm import LLM, SamplingParams, LoraRequest

# # --- 1. Load the BASE model with vLLM and enable LoRA ---
# # The 'enable_lora=True' flag is critical. It prepares the engine to accept LoRA requests.
# print("Loading base model with vLLM...")
# llm = LLM(
#     model="google/gemma-3-1b-it", # Or your chosen base model
#     enable_lora=True,
#     gpu_memory_utilization=0.6 # Adjust as needed
# )
# tokenizer = llm.get_tokenizer()
# print("Model loaded.")


# # --- 2. Load your data from the CSV file ---
# try:
#     df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
#     # Let's just process the first 5 rows for this example
#     prompts = df['body'].head(5).tolist()
#     row_ids = df['row_id'].head(5).tolist()
# except FileNotFoundError:
#     print("Error: 'test.csv' not found.")
#     prompts = []


# # --- 3. Define Sampling Parameters ---
# sampling_params = SamplingParams(
#     temperature=0.8,
#     top_p=0.95,
#     max_tokens=256, # Reduced for a quicker example
# )

# # --- 4. Process the prompts (if any) ---
# if prompts:
#     # Format the prompts using the chat template
#     # Here we assume a simple user role for each body text
#     formatted_prompts = [
#         tokenizer.apply_chat_template(
#             [{"role": "user", "content": prompt}],
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         for prompt in prompts
#     ]

#     # --- 5. Generate output using the LoRA adapter ---
#     # This is the key vLLM step. We create a LoraRequest object.
#     print("Generating responses with LoRA adapter...")
#     outputs = llm.generate(
#         formatted_prompts,
#         sampling_params,
#         lora_request=LoraRequest(
#             lora_name="my_reddit_adapter",          # A unique name you give to this adapter
#             lora_local_files_path="/kaggle/working/grpo_saved_lora" # The path to your saved adapter folder
#         )
#     )
#     print("Generation complete.")

#     # --- 6. Print the results ---
#     for i, output in enumerate(outputs):
#         original_prompt = prompts[i]
#         generated_text = output.outputs[0].text
#         print(f"--- Entry {row_ids[i]} ---")
#         print(f"Original Body:\n{original_prompt}\n")
#         print(f"Generated Output:\n{generated_text.strip()}")
#         print("-" * (len(f"--- Entry {row_ids[i]} ---")) + "\n")

# else:
#     print("No prompts to process.")




