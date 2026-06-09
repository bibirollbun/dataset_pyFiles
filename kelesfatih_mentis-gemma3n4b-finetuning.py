!mkdir /kaggle/tmp


!ls -ld /kaggle/tmp


!apt-get update
!apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y
!git clone https://github.com/ggerganov/llama.cpp /kaggle/tmp/llama.cpp
!cmake /kaggle/tmp/llama.cpp -B /kaggle/tmp/llama.cpp/build \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON
!cmake --build /kaggle/tmp/llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split llama-mtmd-cli
!cp /kaggle/tmp/llama.cpp/build/bin/llama-* /kaggle/tmp/llama.cpp


%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade transformers # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it", # Or "unsloth/gemma-3n-E2B-it"
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)


from transformers import TextStreamer
import gc
# Helper function for inference
def do_gemma_3n_inference(model, messages, max_new_tokens = 128):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )
    # Cleanup to reduce VRAM usage
    del inputs
    torch.cuda.empty_cache()
    gc.collect()


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    use_gradient_checkpointing="unsloth",
    bias = "none",
    random_state = 3407,
)


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


import pandas as pd
from datasets import load_dataset

dataset_psychocounsel = load_dataset("Psychotherapy-LLM/PsychoCounsel-Preference")
df_train = dataset_psychocounsel['train'].to_pandas()

quality_filter = (
        (df_train['chosen_empathy_rating'] == 5) &
        (df_train['chosen_relevance_rating'] == 5) &
        (df_train['chosen_clarity_rating'] >= 4) &
        (df_train['chosen_safety_rating'] == 5) &
        (df_train['chosen_exploration_rating'] >= 4) &
        (df_train['chosen_autonomy_rating'] == 5) &
        (df_train['chosen_staging_rating'] >= 4)
    )
q_filtered_df = df_train[quality_filter].copy()
q_filtered_df = q_filtered_df.drop_duplicates()

def transform_to_conversations(df):
    filtered_df = df[['question', 'chosen']].copy()
    filtered_df = filtered_df.drop_duplicates()
    conversations_list = []
    for _, row in filtered_df.iterrows():
        conversation = [
            {
                "from": "human",
                "value": row['question']
            },
            {
                "from": "gpt",
                "value": row['chosen']
            }
        ]
        conversations_list.append(conversation)
    conversations_df = pd.DataFrame({
        'conversations': conversations_list
    })
    return conversations_df

conversations_df = transform_to_conversations(q_filtered_df)
from datasets import Dataset
dataset = Dataset.from_pandas(conversations_df)


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = None,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)


# @title Show current memory stats
import gc
gc.collect()
torch.cuda.empty_cache()
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title Show final memory and time stats
gc.collect()
torch.cuda.empty_cache()
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


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


model.save_pretrained("gemma-3n")  # Local saving
tokenizer.save_pretrained("gemma-3n")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

# messages = [{
#     "role": "user",
#     "content": [{"type" : "text", "text" : "What is Gemma-3N?",}]
# }]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt = True, # Must add for generation
#     return_tensors = "pt",
#     tokenize = True,
#     return_dict = True,
# ).to("cuda")

# from transformers import TextStreamer
# _ = model.generate(
#     **inputs,
#     max_new_tokens = 128, # Increase for longer outputs!
#     # Recommended Gemma-3 settings!
#     temperature = 1.0, top_p = 0.95, top_k = 64,
#     streamer = TextStreamer(tokenizer, skip_prompt = True),
# )


if True: # Change to True to save finetune!
    model.save_pretrained_merged("/kaggle/tmp/gemma-3N-finetune", tokenizer)


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "hfaccaount", tokenizer,
        token = "token"
    )


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "/kaggle/tmp/gemma-3N-finetune",
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


!python /kaggle/tmp/llama.cpp/convert_hf_to_gguf.py /kaggle/tmp/gemma-3N-finetune \
    --outfile gemma3n-finetune-mentis-model-Q8_0.gguf --outtype q8_0 \
    --split-max-size 50G


from huggingface_hub import login

login(token="token")  # Create one at https://huggingface.co/settings/tokens


from huggingface_hub import create_repo

create_repo("gemma-3N-finetune-mentis", private=True, repo_type="model")


from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/kaggle/working/gemma3n-finetune-mentis-model-Q8_0.gguf",  # local path to .gguf file
    path_in_repo="gemma3n-finetune-mentis-model-Q8_0.gguf",                 # name on the hub
    repo_id="kelesfatih/gemma-3N-finetune-mentis",                 # repo slug
    repo_type="model"                                      # usually "model"
)


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "/kaggle/working/gemma3n-finetune-mentis-model-Q8_0.gguf",
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "account",
        token = "token"
    )

