%%capture
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth
!pip install --upgrade transformers==4.53.2 "huggingface_hub>=0.34.0" "datasets>=3.4.1,<4.0.0"



from unsloth import FastModel
import torch


model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from sklearn.model_selection import train_test_split
import pandas as pd

train_data = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
# 1) split off 20% for validation, stratifying on the rule so each rule remains represented
val_data, eval_data = train_test_split(
    train_data,
    test_size=0.2,
    random_state=42,
    stratify=train_data['rule']
)


from datasets import Dataset
from unsloth.chat_templates import standardize_data_formats
import pandas as pd

# 1) You already have val_data as a pandas.DataFrame:
#    columns: [body, rule, subreddit, …, rule_violation]

# 2) Turn it into a HuggingFace Dataset
val_ds = Dataset.from_pandas(val_data)

# 3) Build the `conversations` field for each example:
def make_conversation(example):
    # 3a) assemble the user‐prompt
    user_text = (
        f"Rule: {example['rule']}\n"
        f"Comment: {example['body']}\n"
        f"positive_example_1 : {example['positive_example_1']}\n"
        f"positive_example_2 : {example['positive_example_2']}\n"
        f"negative_example_1 : {example['negative_example_1']}\n"
        f"negative_example_2 : {example['negative_example_2']}\n"
        "Violates rule?"
    )
    # 3b) the “assistant” response must begin with exactly one space,
    #     so that when tokenized it lines up with the generation prompt
    #     (“ Yes” vs. “ No”)
    assistant_text = "Yes" if example["rule_violation"] == 1 else "No"

    return {
        "conversations": [
            {"role": "user",      "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        # “source” and “score” satisfy standardize_data_formats’s schema:
        "source": "reddit-rule-task",
        "score": 1.0
    }

# 4) Map to add those fields
conv_ds = val_ds.map(make_conversation)

# 5) Now standardize into the internal format
std_ds = standardize_data_formats(conv_ds)

# 6) Finally apply the exact same formatting_prompts_func from the notebook:
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer
          .apply_chat_template(convo,
                               tokenize=False,
                               add_generation_prompt=False)
          .removeprefix("<bos>")
        for convo in convos
    ]
    return { "text": texts }

final_ds = std_ds.map(formatting_prompts_func, batched=True)



final_ds[100]["text"]



# 1. Turn your pandas eval_data into an HF Dataset
eval_ds = Dataset.from_pandas(eval_data)

# 2. Map to conversations
eval_conv = eval_ds.map(make_conversation)

# 3. Standardize
eval_std = standardize_data_formats(eval_conv)

# 4. Apply formatting_prompts_func
eval_final = eval_std.map(formatting_prompts_func, batched=True)



eval_final[100]['text']


import numpy as np
from sklearn.metrics import roc_auc_score

def compute_metrics(eval_preds):
    logits = eval_preds.predictions  # shape (N, seq_len, vocab_size)
    # we know the “Yes” token id:
    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    # extract the logit for “ Yes” at the last generation position:
    preds = logits[:, -1, yes_id]
    probs = torch.sigmoid(torch.tensor(preds)).numpy()
    labels = eval_preds.label_ids.astype(int)  # 1 or 0
    return {"auc": roc_auc_score(labels, probs)}


from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = final_ds,
    eval_dataset  = eval_final,
    args = SFTConfig(
        dataset_text_field         = "text",
        per_device_train_batch_size= 2,
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps= 4,
        warmup_steps               = 5,
        num_train_epochs           = 1,
        learning_rate              = 2e-4,
        logging_steps              = 1,
        optim                      = "adamw_8bit",
        weight_decay               = 0.01,
        lr_scheduler_type          = "linear",
        seed                       = 3407,
        report_to                  = "none",

        # evaluate for _loss_ only, never try to gather logits
        prediction_loss_only       = True,

        # still save checkpoints so training can resume if you want:
        save_strategy              = "steps",
        save_steps                 = 50,
        save_total_limit           = 2,
    )
)



from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")



model.save_pretrained("gemma-3")  # Local saving
tokenizer.save_pretrained("gemma-3")


from kaggle_secrets import UserSecretsClient

if False:
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("HF_MODELS")
    model.push_to_hub("HF_USER_NAME/gemma-3_reddit_4B", token = secret_value_0) # Online saving
    tokenizer.push_to_hub("HF_USER_NAME/gemma-3_reddit_4B", token = secret_value_0) # Online saving


model.save_pretrained_merged("gemma-3-4Bfinetune_vllm_float16", tokenizer)


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "dnouv/gemma-3-4Bfinetune_vllm_float16", tokenizer,
        token = secret_value_0
    )

